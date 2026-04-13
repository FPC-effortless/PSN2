"""T10/T14: PSN2System core.

Fixes applied:
  #1  pulse_error_loss from PhaseController is added to total loss so
      gradients flow through the A-F cycle into node.nu.
  #2  Description length (dl) is computed from actual node compression
      ratio rather than a hardcoded 3.0.
  #5  PSN2LossFamily.compute() is wired into forward_batch; all active
      loss components are computed and summed.
  #8  Grid decoder is spatial: per-cell predictions via a small conv head
      instead of broadcasting one vector across all H×W cells.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .vsa import bundle, normalize, cosine
from .sce import SpikeGate
from .node import NodeBank
from .attractor import AttractorLibrary
from .curiosity import CuriosityEngine, CuriosityGoal
from .motif import MotifLibrary
from .growth import GrowthLedger
from .ers import ExperienceReplaySubstrate
from .phases import PhaseController
from .bonds import BondSystem
from .losses import PSN2LossFamily, active_losses


class SpatialGridDecoder(nn.Module):
    """
    Fix #8: per-cell spatial predictions.
    Takes a global shape vector [B,D], the per-cell embeddings [B,H,W,D],
    and the raw input grid [B,H,W] to produce per-cell logits [B,H,W,vocab].
    The input grid is used directly so unmasked cells can be copied exactly.
    """
    def __init__(self, dim: int, vocab: int):
        super().__init__()
        self.vocab = vocab
        # Fuse global context with local cell embedding
        self.fuse = nn.Linear(dim * 2, dim)
        self.out  = nn.Linear(dim, vocab)

    def forward(self, shape: torch.Tensor, cell_emb: torch.Tensor,
                input_grid: torch.Tensor = None) -> torch.Tensor:
        """
        shape:      [B, D]
        cell_emb:   [B, H, W, D]
        input_grid: [B, H, W]  — raw input token ids (optional, for residual copy)
        returns:    [B, H, W, vocab]
        """
        B, H, W, D = cell_emb.shape
        ctx = shape.unsqueeze(1).unsqueeze(1).expand(B, H, W, D)  # [B,H,W,D]
        fused = torch.cat([ctx, cell_emb], dim=-1)                 # [B,H,W,2D]
        logits = self.out(F.relu(self.fuse(fused)))                # [B,H,W,vocab]

        # Residual copy bias: boost the logit for the input token at each cell.
        # This lets the model learn "copy unmasked, predict masked" naturally.
        if input_grid is not None:
            # One-hot encode input_grid and add as a strong prior
            one_hot = F.one_hot(input_grid, num_classes=self.vocab).float()  # [B,H,W,vocab]
            logits = logits + 2.0 * one_hot  # additive bias toward copying input

        return logits


class PSN2System(nn.Module):
    def __init__(self, dim: int, max_nodes: int, grid_vocab: int, rel_vocab: int,
                 stage: str = "D1"):
        super().__init__()
        self.dim = dim
        self.max_nodes = max_nodes
        self.stage = stage

        # Core subsystems
        self.node_bank   = NodeBank(max_nodes, dim)
        self.spike_gate  = SpikeGate()
        self.attractors  = AttractorLibrary(dim=dim, max_size=2048)
        self.ers         = ExperienceReplaySubstrate(dim=dim)
        self.curiosity   = CuriosityEngine(ers=self.ers)
        self.motifs      = MotifLibrary(dim=dim)
        self.growth      = GrowthLedger()
        self.bond_system = BondSystem(dim=dim)
        self.loss_family = PSN2LossFamily(current_stage=stage)

        # Optional subsystems (activated by stage)
        self.ess         = None
        self.usl         = None
        self.isl         = None
        self.tae_tal     = None
        self.tae_detector = None
        self.tae_replay  = None

        # Grid encoder + Fix #8: spatial decoder
        self.grid_encoder   = nn.Embedding(grid_vocab, dim)
        self.grid_decoder   = SpatialGridDecoder(dim, grid_vocab)

        # Graph encoder/decoder
        self.entity_encoder   = nn.Embedding(rel_vocab, dim)
        self.relation_encoder = nn.Embedding(32, dim)
        # Entity decoder: uses relational neighborhood context around the masked slot
        # Input: [neighbor_context (dim) + global_shape (dim)] -> rel_vocab logits
        self.entity_decoder   = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, rel_vocab),
        )
        # GAP 3g: relation_decoder was missing
        self.relation_decoder = nn.Linear(dim, 32)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """grid: [B, H, W] -> [B, D] mean-pooled embedding."""
        emb = self.grid_encoder(grid)          # [B, H, W, D]
        return emb.mean(dim=[1, 2])            # [B, D]

    def encode_graph(self, entities: torch.Tensor,
                     relations: torch.Tensor) -> torch.Tensor:
        """
        entities:  [B, N_e]
        relations: [B, N_r, 3]  (a, r, b)
        Returns [B, D] shape vector.
        """
        ent_emb = self.entity_encoder(entities).mean(dim=1)   # [B, D]
        # relations[:,:,1] are relation type indices
        rel_type = relations[:, :, 1].clamp(min=0, max=31)    # [B, N_r]
        rel_emb = self.relation_encoder(rel_type).mean(dim=1)  # [B, D]
        return normalize(ent_emb + rel_emb)

    def encode_graph_neighborhood(self, entities: torch.Tensor,
                                   relations: torch.Tensor,
                                   masked_idx: torch.Tensor) -> torch.Tensor:
        """
        Build a neighborhood context vector for the masked entity slot.
        Aggregates embeddings of entities that appear in relations with the masked slot,
        weighted by the relation type embedding.

        entities:   [B, N_e]
        relations:  [B, N_r, 3]  (a, r, b) — all long tensors
        masked_idx: [B]           — which entity slot is masked
        Returns:    [B, D]
        """
        B, N_e = entities.shape
        N_r = relations.shape[1]
        device = entities.device

        ent_emb = self.entity_encoder(entities)  # [B, N_e, D]
        rel_type = relations[:, :, 1].clamp(min=0, max=31)  # [B, N_r]
        rel_emb = self.relation_encoder(rel_type)            # [B, N_r, D]

        # Vectorized: for each relation triple (a, r, b), check if a==masked or b==masked
        # relations: [B, N_r, 3]
        rel_a = relations[:, :, 0]  # [B, N_r] — source entity index
        rel_b = relations[:, :, 2]  # [B, N_r] — target entity index
        m = masked_idx.unsqueeze(1).expand(B, N_r)  # [B, N_r]

        # Masks: which relations involve the masked entity
        is_source = (rel_a == m)  # [B, N_r] — masked is source, neighbor is rel_b
        is_target = (rel_b == m)  # [B, N_r] — masked is target, neighbor is rel_a

        # Gather neighbor entity embeddings
        # When masked is source: neighbor index = rel_b
        # When masked is target: neighbor index = rel_a
        neighbor_idx_src = rel_b.clamp(0, N_e - 1)  # [B, N_r]
        neighbor_idx_tgt = rel_a.clamp(0, N_e - 1)  # [B, N_r]

        # Gather: [B, N_r, D]
        neighbor_emb_src = ent_emb.gather(
            1, neighbor_idx_src.unsqueeze(-1).expand(B, N_r, self.dim)
        )
        neighbor_emb_tgt = ent_emb.gather(
            1, neighbor_idx_tgt.unsqueeze(-1).expand(B, N_r, self.dim)
        )

        # Select neighbor embedding based on which role the masked entity plays
        # [B, N_r, D]: use src neighbor when masked is source, tgt neighbor otherwise
        is_src_mask = is_source.unsqueeze(-1).float()   # [B, N_r, 1]
        is_tgt_mask = is_target.unsqueeze(-1).float()   # [B, N_r, 1]
        neighbor_emb = is_src_mask * neighbor_emb_src + is_tgt_mask * neighbor_emb_tgt  # [B, N_r, D]

        # Add relation embedding to neighbor context
        involved = (is_source | is_target).unsqueeze(-1).float()  # [B, N_r, 1]
        context = (neighbor_emb + rel_emb) * involved  # [B, N_r, D], zero for uninvolved

        # Sum and normalize by count of involved relations
        count = involved.sum(dim=1).clamp(min=1.0)  # [B, 1]
        neighborhood = context.sum(dim=1) / count   # [B, D]

        # Fallback: if no relations involve the masked entity, use global entity mean
        has_neighbors = (is_source | is_target).any(dim=1)  # [B]
        no_neighbors = ~has_neighbors
        if no_neighbors.any():
            global_mean = ent_emb[no_neighbors].mean(dim=1)
            neighborhood[no_neighbors] = global_mean

        return normalize(neighborhood)  # [B, D]

    # ------------------------------------------------------------------
    # GAP 3e: _make_controller
    # ------------------------------------------------------------------
    def _make_controller(self, phase: str, budget: int = 32) -> PhaseController:
        """Create a PhaseController for one forward pass."""
        controller = PhaseController(
            node_bank=self.node_bank,
            budget=budget,
            bond_system=self.bond_system,
            ess=self.ess,
        )
        controller.active_regime = phase
        return controller

    # ------------------------------------------------------------------
    # GAP 3f: _node_repr
    # ------------------------------------------------------------------
    def _node_repr(self, shape: torch.Tensor) -> torch.Tensor:
        """
        Returns a [B, D] representation by averaging active node activations
        weighted by cosine similarity to the shape.
        shape: [B, D] or [D]
        """
        if shape.dim() == 1:
            shape = shape.unsqueeze(0)  # [1, D]
        B, D = shape.shape
        active_mask = self.node_bank.active.bool()
        if not active_mask.any():
            return shape
        active_nu = self.node_bank.nu[active_mask]          # [N_active, D]
        # Cosine similarity: [B, N_active]
        shape_norm = F.normalize(shape, dim=-1)
        nu_norm = F.normalize(active_nu.detach(), dim=-1)
        sims = torch.matmul(shape_norm, nu_norm.t())        # [B, N_active]
        weights = torch.softmax(sims, dim=-1)               # [B, N_active]
        repr_vec = torch.matmul(weights, active_nu)         # [B, D]
        return normalize(repr_vec)

    # ------------------------------------------------------------------
    # GAP 3d: forward_batch — clean rewrite
    # ------------------------------------------------------------------
    def forward_batch(self, batch: dict, phase: str = "perceptive") -> dict:
        btype = batch["type"]
        device = self.node_bank.nu.device

        # GAP 3e: create controller with B_max Lite = 32
        controller = self._make_controller(phase, budget=32)

        if btype == "arc":
            grid = batch["input_grid"]
            target = batch["target_grid"]
            shape = self.encode_grid(grid)

            # ISL modulatory input (D4+)
            modulatory = None
            if self.isl is not None and controller.committed_shape is not None:
                modulatory = self.isl.run(
                    controller.committed_shape,
                    pulse_remaining=controller.budget,
                    budget_max=controller.max_budget,
                    regime=phase,
                    uncertainty=float(self.node_bank.e.mean().item()),
                )

            committed_shape = controller.run_pulse(external_input=shape, modulatory_input=modulatory, regime=phase)
            if committed_shape is not None:
                # committed_shape may be [D] (collapsed by phase_f mean); restore batch dim
                if committed_shape.dim() == 1:
                    committed_shape = committed_shape.unsqueeze(0).expand(shape.size(0), -1)
                active_shape = committed_shape
            else:
                active_shape = shape

            # Fix #10: TAE motif replay in perceptive regime
            if self.tae_replay is not None and phase == "perceptive":
                replayed = self.tae_replay.attempt_replay(
                    shape.mean(dim=0), regime=phase,
                    verifier_fn=None,
                )
                if replayed is not None:
                    active_shape = replayed.unsqueeze(0).expand_as(active_shape)

            node_repr = self._node_repr(active_shape)
            # Spatial grid decode: need per-cell embeddings
            cell_emb = self.grid_encoder(grid)              # [B, H, W, D]
            pred_logits = self.grid_decoder(node_repr, cell_emb, input_grid=grid)  # [B, H, W, vocab]

            budget_fraction = controller.budget_fraction_used
            loss_pred = F.cross_entropy(pred_logits.reshape(-1, pred_logits.size(-1)), target.reshape(-1))
            target_shape = self.encode_grid(target)
            loss_shape = 1.0 - cosine(active_shape.mean(dim=0), target_shape.mean(dim=0))
            spike_mask = self.node_bank.spike_mask(phase, budget_fraction_used=budget_fraction)
            loss_spike = spike_mask.mean()

            # Fix #11: apply ESS delta_theta to commit threshold (modulate loss_shape)
            ess_delta = 0.0
            if self.ess is not None:
                ess_delta = self.ess.get_threshold_delta(id(active_shape))
            loss_shape = loss_shape * (1.0 + ess_delta)

            # Fix #14: wire loss family — compute all active losses
            loss_vsa = self.loss_family.L_vsa(active_shape, target_shape)
            loss_compact = torch.tensor(float(len(self.attractors)), device=device) * 1e-5
            loss_components = {
                "L_error": loss_pred,
                "L_shape": loss_shape if isinstance(loss_shape, torch.Tensor) else torch.tensor(loss_shape),
                "L_vsa": loss_vsa,
                "L_compact": loss_compact,
            }
            family_loss = self.loss_family.compute(loss_components)

            total_loss = family_loss + 0.01 * loss_spike

            # Add pulse_error_loss from controller
            if controller.pulse_error_loss is not None:
                total_loss = total_loss + 0.01 * controller.pulse_error_loss

            # GAP 3d: increment node ages
            if hasattr(self.node_bank, "increment_age"):
                self.node_bank.increment_age()

            return {
                "loss": total_loss,
                "loss_pred": loss_pred.detach(),
                "loss_shape": loss_shape.detach(),
                "loss_spike": loss_spike.detach(),
                "shape": active_shape.detach(),
                "pred": pred_logits.detach(),
            }

        elif btype == "graph":
            entities = batch["entities"]
            relations = batch["relations"]
            target_entity = batch["target_entity"]
            target_relation = batch["target_relation"]
            masked_entity_idx = batch.get("masked_entity_idx", torch.zeros_like(target_entity))

            shape = self.encode_graph(entities, relations)
            committed_shape = controller.run_pulse(external_input=shape, regime=phase)
            if committed_shape is not None:
                if committed_shape.dim() == 1:
                    committed_shape = committed_shape.unsqueeze(0).expand(shape.size(0), -1)
                active_shape = committed_shape
            else:
                active_shape = shape

            budget_fraction = controller.budget_fraction_used
            node_repr = self._node_repr(active_shape)

            # Use relational neighborhood context to predict the masked entity
            # This gives the decoder the actual signal: which entities neighbor the masked slot
            neighborhood = self.encode_graph_neighborhood(entities, relations, masked_entity_idx)

            # Concatenate neighborhood context with global shape for entity prediction
            entity_input = torch.cat([neighborhood, node_repr], dim=-1)  # [B, 2D]
            entity_logits = self.entity_decoder(entity_input)

            # Relation prediction from global shape
            relation_logits = self.relation_decoder(node_repr)

            loss_entity = F.cross_entropy(entity_logits, target_entity)
            loss_relation = F.cross_entropy(relation_logits, target_relation)

            # Shape loss: encourage active_shape to match a target encoding
            with torch.no_grad():
                pred_entity_ids = entity_logits.argmax(dim=-1)
                pred_entities = pred_entity_ids.unsqueeze(1).expand_as(entities)
                target_shape = self.encode_graph(pred_entities, relations).detach()
            loss_shape = 1.0 - F.cosine_similarity(active_shape, target_shape, dim=-1).mean()

            spike_mask = self.node_bank.spike_mask(phase, budget_fraction_used=budget_fraction)
            loss_spike = spike_mask.mean()

            loss = loss_entity + 0.5 * loss_relation + 0.1 * loss_shape + 0.01 * loss_spike

            if controller.pulse_error_loss is not None:
                loss = loss + 0.01 * controller.pulse_error_loss

            if hasattr(self.node_bank, "increment_age"):
                self.node_bank.increment_age()

            return {
                "loss": loss,
                "loss_pred": (loss_entity + loss_relation).detach(),
                "loss_shape": loss_shape.detach(),
                "loss_spike": loss_spike.detach(),
                "shape": active_shape.detach(),
                "pred": entity_logits.detach(),
            }
        else:
            raise ValueError(f"Unknown batch type: {btype}")

    def maybe_grow(self, global_step: int, error_value: float):
        """T07: SGP — all four operations: spawn, prune, merge, stability gate."""
        # Fix #9: compute actual description length from attractor compression ratio
        n_active = int(self.node_bank.active.sum().item())
        n_attractors = len(self.attractors)
        dl = float(n_active) / max(n_attractors, 1)  # high when nodes >> attractors

        # Operation 1: spawn
        self.growth.maybe_spawn_node(self.node_bank, global_step, self.attractors)

        # Fix: Operations 2, 3, 4 — prune/merge every 50 steps for balanced growth
        # Too frequent (every 10) causes collapse, too rare (every 500) blocks spawn via I-24
        if global_step % 50 == 0 and global_step > 0:
            self.growth.maybe_prune_nodes(self.node_bank, global_step)
            self.growth.maybe_merge_nodes(self.node_bank, global_step)

        # Stability gate evaluation
        self.growth.evaluate_stability_gates(self.node_bank, global_step)

        # Curiosity goal generation
        if error_value > 0.6:
            dummy_vsa = normalize(torch.randn(self.dim, device=self.node_bank.nu.device))
            self.curiosity.detect_and_generate(
                node_id=global_step,
                error=error_value,
                dl=dl,                  # Fix #9: actual DL, not hardcoded 3.0
                budget_fraction=0.2,
                committed=False,
                target_vsa=dummy_vsa,
            )
            self.growth.maybe_expand_attractor(self.attractors, dummy_vsa, global_step)

        # Fix #12: age goals and retire stale ones
        self.curiosity.tick_episode()

        # Fix: Promote ERS memories more frequently to prevent Working tier saturation
        # Working tier capacity = 128, promote every 100 steps (not just in maybe_grow)
        if global_step % 100 == 0:
            self.ers.attempt_promotions(session_end=False)

        # Bond decay, motif promotion
        self.bond_system.pulse_decay()
        self.ers.attempt_promotions()
        self.motifs.promote_eligible()

    def activate_ess(self):
        """Activate Emotional Shape System (D3+)."""
        from psn2.ess.emotional_shapes import EmotionalShapeSystem
        self.ess = EmotionalShapeSystem(self.dim)

    def activate_usl(self, vocab_size: int = 1000):
        """Activate USL Codec and ISL (D4+). GAP 3c: max_size=8192 per PRD Lite config."""
        from psn2.usl.lal import LexicalAttractorLibrary
        from psn2.usl.codec import USLCodec
        from psn2.usl.isl import InnerSpeechLoop
        lal = LexicalAttractorLibrary(self.dim, max_size=8192)
        self.usl = USLCodec(self.dim, vocab_size, lal)
        self.isl = InnerSpeechLoop(self.usl, self.ers, self.dim)

    def activate_tae(self):
        """Activate Temporal Abstraction Engine (D5+)."""
        from psn2.tae.tal import TemporalAttractorLibrary
        from psn2.tae.motif_detector import MotifDetector
        from psn2.tae.motif_replay import MotifReplay
        self.tae_tal = TemporalAttractorLibrary(self.dim)
        self.tae_detector = MotifDetector(self.dim, self.tae_tal)
        self.tae_replay = MotifReplay(self.tae_tal)

    def state_dict_full(self) -> dict:
        """GAP 3a: include ERS state in serialization."""
        return {
            "model": self.state_dict(),
            "attractors": self.attractors.state_dict(),
            "attractor_utility": self.attractors.utility,
            "curiosity": self.curiosity.state_dict(),
            "motifs": self.motifs.state_dict(),
            "growth": self.growth.state_dict(),
            "bonds": self.bond_system.state_dict(),
            "ers": self.ers.state_dict(),
            "node_e": self.node_bank.e.clone(),
            "node_tau": self.node_bank.tau.clone(),
            "node_sigma": self.node_bank.sigma.clone(),
            "node_active": self.node_bank.active.clone(),
            "stage": self.stage,
        }

    def load_state_dict_full(self, state: dict):
        self.load_state_dict(state["model"], strict=False)
        # Support both old list format and new dict format for attractors
        attr_state = state.get("attractors", {})
        if isinstance(attr_state, dict):
            self.attractors.load_state_dict(attr_state)
        else:
            # Legacy: list of vectors
            self.attractors.codebook = [torch.tensor(v) for v in attr_state]
            self.attractors.utility = state.get("attractor_utility", [0.5] * len(self.attractors.codebook))
        self.curiosity.load_state_dict(state.get("curiosity", []))
        self.motifs.load_state_dict(state.get("motifs", []))
        self.growth.load_state_dict(state.get("growth", {}))
        self.bond_system.load_state_dict(state.get("bonds", {}))
        # GAP 3a: restore ERS state
        if "ers" in state:
            self.ers.load_state_dict(state["ers"])
        if "node_e" in state:
            self.node_bank.e.copy_(state["node_e"])
        if "node_tau" in state:
            self.node_bank.tau.copy_(state["node_tau"])
        if "node_sigma" in state:
            self.node_bank.sigma.copy_(state["node_sigma"])
        if "node_active" in state:
            self.node_bank.active.copy_(state["node_active"])
        if "stage" in state:
            self.stage = state["stage"]
            self.loss_family.current_stage = self.stage
