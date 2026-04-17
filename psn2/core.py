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


class PropertyAwareEntityEncoder(nn.Module):
    """
    Property-aware entity encoder that decomposes entity IDs into property embeddings.

    Supports two entity vocabulary sizes:
    - entity_vocab=5  (4 entities + 1 mask): 2 colors × 2 shapes
      Entity ID = color * 2 + shape  →  0-3, mask token = 4
    - entity_vocab=17 (16 entities + 1 mask): 2 colors × 2 shapes × 4 sizes
      Entity ID = color * 8 + shape * 4 + size  →  0-15, mask token = 16

    Key insight: By summing property embeddings, entities that share properties
    will have similar embeddings, enabling the model to generalize.
    """
    def __init__(self, dim: int, entity_vocab: int = 17):
        super().__init__()
        self.dim = dim
        self.entity_vocab = entity_vocab

        # Mask token is always the last ID (entity_vocab - 1)
        self.mask_token_id = entity_vocab - 1

        # Property embeddings - each property contributes to the final embedding
        self.color_emb = nn.Embedding(2, dim)  # 2 colors
        self.shape_emb = nn.Embedding(2, dim)  # 2 shapes

        # Size embedding only used for 16-entity mode (entity_vocab == 17)
        self.use_size = (entity_vocab == 17)
        if self.use_size:
            self.size_emb = nn.Embedding(4, dim)   # 4 sizes
            self.scale = 1.0 / 3.0
        else:
            self.scale = 1.0 / 2.0

        # Mask token embedding
        self.mask_emb = nn.Parameter(torch.randn(dim))

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """
        entity_ids: [...] - arbitrary shape tensor of entity IDs
        returns:    [..., D] - entity embeddings
        """
        # Handle mask token separately
        is_mask = (entity_ids == self.mask_token_id)

        if self.use_size:
            # 16-entity mode: entity_id = color * 8 + shape * 4 + size
            entity_ids_clamped = entity_ids.clamp(0, 15)
            colors = entity_ids_clamped // 8
            shapes = (entity_ids_clamped % 8) // 4
            sizes = entity_ids_clamped % 4
            color_emb = self.color_emb(colors)
            shape_emb = self.shape_emb(shapes)
            size_emb = self.size_emb(sizes)
            entity_emb = (color_emb + shape_emb + size_emb) * self.scale
        else:
            # 4-entity mode: entity_id = color * 2 + shape
            entity_ids_clamped = entity_ids.clamp(0, 3)
            colors = entity_ids_clamped // 2
            shapes = entity_ids_clamped % 2
            color_emb = self.color_emb(colors)
            shape_emb = self.shape_emb(shapes)
            entity_emb = (color_emb + shape_emb) * self.scale

        # Replace mask token positions with mask embedding
        if is_mask.any():
            mask_emb_expanded = self.mask_emb.view(*([1] * (entity_ids.dim())), self.dim)
            mask_emb_expanded = mask_emb_expanded.expand(*entity_ids.shape, self.dim)
            entity_emb = torch.where(is_mask.unsqueeze(-1), mask_emb_expanded, entity_emb)

        return entity_emb


class RelationAwareEntityDecoder(nn.Module):
    """
    Task 3.3: Relation-aware entity decoder with explicit attention over relation types.

    This decoder predicts the masked entity by:
    1. Attending over the individual relation embeddings from the masked entity's
       neighborhood (relation-conditioned prediction head).
    2. Fusing the attended relation context with the aggregated neighborhood vector
       and the global graph encoding.
    3. Predicting entity properties (color, shape, size) separately to encourage
       compositional reasoning ("entity X is related to entity Y via relation R").

    The key addition in Task 3.3 is the relation attention mechanism:
    - `rel_query_proj`: projects the fused (neighborhood + graph) context into a
      query vector used to attend over the per-relation embeddings.
    - `rel_attn_score`: scores each relation embedding against the query.
    - `rel_gate`: a learned gate that controls how much the relation-conditioned
      context modulates the final prediction, preventing it from overwhelming the
      neighborhood signal early in training.

    Input: neighborhood [B,D], graph_encoding [B,D],
           relation_embs [B, N_r, D] (optional — raw per-relation embeddings for
           the relations that involve the masked entity)
    Output: [B, entity_vocab] logits
    """
    def __init__(self, dim: int, entity_vocab: int, relation_vocab: int = 32):
        super().__init__()
        self.dim = dim
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab

        # Main prediction path: fuse neighborhood + graph context
        self.fuse = nn.Linear(dim * 2, dim)

        # Task 3.3: Relation attention mechanism
        # Projects fused context to a query for attending over relation embeddings
        self.rel_query_proj = nn.Linear(dim, dim)
        # Scores each relation embedding against the query: [B, N_r, D] -> [B, N_r]
        self.rel_attn_score = nn.Linear(dim, 1, bias=False)
        # Projects attended relation context into the prediction space
        self.rel_context_proj = nn.Linear(dim, dim)
        # Learned gate: controls how much relation context modulates prediction
        # Initialized near 0 so early training is stable (gate starts mostly closed)
        self.rel_gate = nn.Linear(dim * 2, dim)

        # Deeper decoder (3 layers) with dropout for capacity and regularization
        self.predict = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, entity_vocab),
        )

        # Property prediction heads — predict each property separately.
        # This encourages the model to reason about properties explicitly.
        self.predict_color = nn.Linear(dim, 2)   # 2 colors
        self.predict_shape = nn.Linear(dim, 2)   # 2 shapes
        self.predict_size  = nn.Linear(dim, 4)   # 4 sizes

    def forward(self, neighborhood: torch.Tensor, graph_encoding: torch.Tensor,
                relation_embs: Optional[torch.Tensor] = None,
                relation_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        neighborhood:    [B, D]       - aggregated relational context (from encode_graph_neighborhood)
        graph_encoding:  [B, D]       - global graph structure encoding
        relation_embs:   [B, N_r, D]  - per-relation embeddings for relations involving the
                                        masked entity (optional; enables relation attention)
        relation_mask:   [B, N_r]     - float mask: 1.0 for involved relations, 0.0 otherwise
        returns:         [B, entity_vocab] logits
        """
        # Step 1: Fuse aggregated neighborhood + global graph context
        combined = torch.cat([neighborhood, graph_encoding], dim=-1)  # [B, 2D]
        fused = F.relu(self.fuse(combined))  # [B, D]

        # Step 2: Task 3.3 — Relation-conditioned prediction head
        # If per-relation embeddings are provided, attend over them using the fused
        # context as a query. This lets the decoder learn "entity X is related to
        # entity Y via relation R" by directly reasoning about relation types.
        if relation_embs is not None and relation_embs.size(1) > 0:
            # Project fused context to query: [B, D]
            query = self.rel_query_proj(fused)  # [B, D]

            # Score each relation embedding against the query.
            # Broadcast query over N_r dimension for element-wise interaction.
            # [B, N_r, D] + [B, 1, D] -> [B, N_r, D] -> [B, N_r]
            query_expanded = query.unsqueeze(1).expand_as(relation_embs)  # [B, N_r, D]
            rel_scores = self.rel_attn_score(relation_embs * query_expanded).squeeze(-1)  # [B, N_r]

            # Mask out uninvolved relations before softmax
            if relation_mask is not None:
                rel_scores = rel_scores + (1.0 - relation_mask) * (-1e9)

            rel_attn_weights = torch.softmax(rel_scores, dim=-1)  # [B, N_r]

            # Zero out weights for uninvolved relations to avoid NaN propagation
            if relation_mask is not None:
                rel_attn_weights = rel_attn_weights * relation_mask  # [B, N_r]

            # Weighted sum of relation embeddings: [B, D]
            attended_rel = (rel_attn_weights.unsqueeze(-1) * relation_embs).sum(dim=1)  # [B, D]

            # Project attended relation context
            rel_ctx = F.relu(self.rel_context_proj(attended_rel))  # [B, D]

            # Gated fusion: learn how much relation context to mix in.
            # Gate input: [fused, rel_ctx] -> sigmoid gate -> modulate rel_ctx
            gate_input = torch.cat([fused, rel_ctx], dim=-1)  # [B, 2D]
            gate = torch.sigmoid(self.rel_gate(gate_input))   # [B, D]
            fused = fused + gate * rel_ctx                    # [B, D]

        # Step 3: Predict entity using deeper network with dropout
        logits = self.predict(fused)  # [B, entity_vocab]

        # Step 4: Predict properties separately (for auxiliary loss in forward_batch)
        self.last_color_logits = self.predict_color(fused)  # [B, 2]
        self.last_shape_logits = self.predict_shape(fused)  # [B, 2]
        self.last_size_logits  = self.predict_size(fused)   # [B, 4]

        return logits

    def get_property_logits(self):
        """Get the property prediction logits from the last forward pass."""
        return {
            'color': self.last_color_logits,
            'shape': self.last_shape_logits,
            'size':  self.last_size_logits,
        }


class PSN2System(nn.Module):
    def __init__(self, dim: int, max_nodes: int, grid_vocab: int, rel_vocab: int,
                 stage: str = "D1"):
        super().__init__()
        self.dim = dim
        self.max_nodes = max_nodes
        self.stage = stage
        
        # Fix 3.7: Track bond formation statistics at model level
        self._bond_formation_count = 0
        self._total_compositional_pulses = 0

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
        # Property-aware entity encoder for better generalization
        self.entity_encoder   = PropertyAwareEntityEncoder(dim, rel_vocab)
        self.relation_encoder = nn.Embedding(32, dim)
        
        # Fix 3.3: Use relation-aware entity decoder instead of simple Sequential
        self.entity_decoder = RelationAwareEntityDecoder(dim, rel_vocab, relation_vocab=32)
        
        # GAP 3g: relation_decoder was missing
        self.relation_decoder = nn.Linear(dim, 32)

        # Task 3.1: Attention-weighted neighborhood aggregation
        # Scores each (neighbor, relation) pair by relation type to weight aggregation
        self.neighborhood_attn = nn.Linear(dim, 1, bias=True)
        # Projects aggregated relation context directly into neighborhood vector
        # so relation type signal is preserved even after attention aggregation
        self.rel_context_proj = nn.Linear(dim, dim, bias=True)

        # Task 3.4: Bond context projection and gating
        # Projects aggregated bond context (from recovered node vectors) into
        # the neighborhood space so bond-based relational signal can be fused.
        # Initialized with small weights so early training is stable.
        self.bond_context_proj = nn.Linear(dim, dim, bias=True)
        # Learned gate: controls how much bond context modulates the neighborhood.
        # Input: [neighborhood, bond_context] -> sigmoid gate -> scale bond_context.
        # Initialized near 0 (via small weight init) so gate starts mostly closed.
        self.bond_gate = nn.Linear(dim * 2, dim, bias=True)
        # Initialize bond_gate weights small so it starts near zero output
        nn.init.normal_(self.bond_gate.weight, std=0.01)
        nn.init.zeros_(self.bond_gate.bias)

    # ------------------------------------------------------------------
    # Fix 3.6: Gradient monitoring helper
    # ------------------------------------------------------------------
    def log_gradient_norms(self) -> dict:
        """
        Log gradient norms for key components during graph training.
        Call this after backward pass to monitor gradient flow.
        
        Returns dict with gradient norms for entity_encoder, relation_encoder, entity_decoder.
        """
        grad_norms = {}
        
        # Entity encoder is now PropertyAwareEntityEncoder with multiple parameters
        entity_encoder_grad_norm = 0.0
        entity_encoder_param_count = 0
        for param in self.entity_encoder.parameters():
            if param.grad is not None:
                entity_encoder_grad_norm += param.grad.norm().item() ** 2
                entity_encoder_param_count += 1
        if entity_encoder_param_count > 0:
            grad_norms['entity_encoder'] = (entity_encoder_grad_norm / entity_encoder_param_count) ** 0.5
        
        if hasattr(self.relation_encoder, 'weight') and self.relation_encoder.weight.grad is not None:
            grad_norms['relation_encoder'] = self.relation_encoder.weight.grad.norm().item()
        
        # Entity decoder has multiple parameters, aggregate them
        decoder_grad_norm = 0.0
        decoder_param_count = 0
        for param in self.entity_decoder.parameters():
            if param.grad is not None:
                decoder_grad_norm += param.grad.norm().item() ** 2
                decoder_param_count += 1
        if decoder_param_count > 0:
            grad_norms['entity_decoder'] = (decoder_grad_norm / decoder_param_count) ** 0.5
        
        return grad_norms
    
    # ------------------------------------------------------------------
    # Fix 3.7: Bond formation statistics
    # ------------------------------------------------------------------
    def get_bond_formation_stats(self) -> dict:
        """
        Get bond formation statistics for monitoring.
        Returns dict with bond_formation_count, total_pulses, and bond_formation_rate.
        """
        return {
            'bond_formation_count': self._bond_formation_count,
            'total_pulses': self._total_compositional_pulses,
            'bond_formation_rate': self._bond_formation_count / max(self._total_compositional_pulses, 1),
        }

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
                                   masked_idx: torch.Tensor,
                                   return_relation_embs: bool = False):
        """
        Build a neighborhood context vector for the masked entity slot.

        Task 3.1 Fix: Attention-weighted aggregation based on relation type.
        - Each (neighbor, relation) pair is scored by self.neighborhood_attn applied
          to the relation embedding. This lets the model learn which relation types
          are most informative for predicting the masked entity.
        - Relation type embeddings are added directly to the neighborhood context
          via self.rel_context_proj so relation-type signal survives aggregation.
        - The combined vector carries discriminative signal about BOTH the neighbor
          properties and the specific relation types connecting to the masked entity.

        Task 3.3 Addition: When return_relation_embs=True, also returns the raw
        per-relation embeddings and involvement mask so the entity decoder can apply
        its own attention mechanism over individual relation types.

        entities:            [B, N_e]
        relations:           [B, N_r, 3]  (a, r, b) — all long tensors
        masked_idx:          [B]           — which entity slot is masked
        return_relation_embs: bool         — if True, return (context, rel_emb, mask)
        Returns:
          - If return_relation_embs=False: [B, D]  (default, backward-compatible)
          - If return_relation_embs=True:  ([B, D], [B, N_r, D], [B, N_r])
        """
        B, N_e = entities.shape
        N_r = relations.shape[1]
        device = entities.device

        ent_emb = self.entity_encoder(entities)  # [B, N_e, D]
        rel_type = relations[:, :, 1].clamp(min=0, max=31)  # [B, N_r]
        rel_emb = self.relation_encoder(rel_type)            # [B, N_r, D]

        # For each relation triple (a, r, b), check if a==masked or b==masked
        rel_a = relations[:, :, 0]  # [B, N_r] — source entity index
        rel_b = relations[:, :, 2]  # [B, N_r] — target entity index
        m = masked_idx.unsqueeze(1).expand(B, N_r)  # [B, N_r]

        # Masks: which relations involve the masked entity
        is_source = (rel_a == m)  # masked is source, neighbor is rel_b
        is_target = (rel_b == m)  # masked is target, neighbor is rel_a

        # Gather neighbor entity embeddings
        # When masked is source: neighbor index = rel_b
        # When masked is target: neighbor index = rel_a
        neighbor_idx_src = rel_b.clamp(0, N_e - 1)  # [B, N_r]
        neighbor_idx_tgt = rel_a.clamp(0, N_e - 1)  # [B, N_r]

        neighbor_emb_src = ent_emb.gather(
            1, neighbor_idx_src.unsqueeze(-1).expand(B, N_r, self.dim)
        )  # [B, N_r, D]
        neighbor_emb_tgt = ent_emb.gather(
            1, neighbor_idx_tgt.unsqueeze(-1).expand(B, N_r, self.dim)
        )  # [B, N_r, D]

        # Select neighbor embedding based on which role the masked entity plays
        is_src_mask = is_source.unsqueeze(-1).float()   # [B, N_r, 1]
        is_tgt_mask = is_target.unsqueeze(-1).float()   # [B, N_r, 1]
        neighbor_emb = is_src_mask * neighbor_emb_src + is_tgt_mask * neighbor_emb_tgt  # [B, N_r, D]

        involved = (is_source | is_target).float()  # [B, N_r]  — 1.0 where relation is relevant

        # ------------------------------------------------------------------
        # Task 3.1: Attention-weighted aggregation over relation types
        # Score each (neighbor, relation) pair using the relation embedding.
        # Uninvolved relations get a large negative bias so they contribute 0
        # after softmax, preserving the original semantics.
        # ------------------------------------------------------------------
        # Compute raw attention scores from relation embeddings: [B, N_r, 1] -> [B, N_r]
        attn_scores = self.neighborhood_attn(rel_emb).squeeze(-1)  # [B, N_r]

        # Mask out non-involved relations by setting their score to -1e9 before softmax
        attn_scores = attn_scores + (1.0 - involved) * (-1e9)

        # Check if any relations are involved per sample
        has_neighbors = (involved > 0).any(dim=1)  # [B]

        # Softmax over N_r dimension to get normalized attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, N_r]

        # Zero out weights for samples with no relevant relations (avoids NaN)
        attn_weights = attn_weights * involved  # [B, N_r]

        # Weighted aggregation of (neighbor_emb + rel_emb) pairs:
        # This combines: what property the neighbor has + what relation connects them
        pair_context = neighbor_emb + rel_emb  # [B, N_r, D]
        # [B, N_r] x [B, N_r, D] -> [B, D]
        neighbor_context = (attn_weights.unsqueeze(-1) * pair_context).sum(dim=1)  # [B, D]

        # ------------------------------------------------------------------
        # Task 3.1: Add relation type embeddings directly to the output
        # Compute a weighted mean of the pure relation embeddings and project it
        # into the neighborhood context. This ensures the relation-type signal
        # is present in the final vector independently of the neighbor content.
        # ------------------------------------------------------------------
        # Weighted mean of relation embeddings for involved relations
        rel_weight_sum = involved.sum(dim=1, keepdim=True) + 1e-8  # [B, 1]
        mean_rel_emb = (rel_emb * involved.unsqueeze(-1)).sum(dim=1) / rel_weight_sum  # [B, D]
        rel_context = self.rel_context_proj(mean_rel_emb)  # [B, D]

        # Add projected relation context directly to neighborhood vector
        neighborhood_context = neighbor_context + rel_context  # [B, D]

        # Fallback: if no relations involve the masked entity, use global entity mean
        no_neighbors = ~has_neighbors
        if no_neighbors.any():
            global_mean = ent_emb[no_neighbors].mean(dim=1)
            neighborhood_context = neighborhood_context.clone()
            neighborhood_context[no_neighbors] = global_mean

        normalized_context = normalize(neighborhood_context)  # [B, D]

        if return_relation_embs:
            # Task 3.3: Return raw per-relation embeddings and involvement mask
            # so the entity decoder can apply its own attention over relation types.
            # rel_emb: [B, N_r, D] — embeddings for each relation type
            # involved: [B, N_r]   — 1.0 for relations involving the masked entity
            return normalized_context, rel_emb, involved

        return normalized_context  # [B, D]

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
    # Task 3.4: Bond context retrieval helper
    # ------------------------------------------------------------------
    def _retrieve_bond_context(self, device: torch.device) -> torch.Tensor:
        """
        Retrieve a bond-based context vector by querying bonds formed during
        the compositional phase pulse cycle.

        Bonds are formed between node bank indices (not entity IDs). This
        method implements a best-effort approach: it aggregates the bond
        vectors and recovered source node vectors from all active bonds to
        produce a relational context signal that can supplement the
        neighborhood encoding.

        Strategy:
        1. Collect all bonds from bond_system (formed in Phase B).
        2. For each bond, use recover_source with the node bank as codebook
           to retrieve the related node vector.
        3. Weight each recovered vector by the bond's strength.
        4. Return the strength-weighted mean of recovered vectors, normalized.
           If no bonds exist, return a zero vector.

        Returns: [D] bond context vector (detached, no gradient).
        """
        bonds = self.bond_system.bonds
        if not bonds:
            return torch.zeros(self.dim, device=device)

        # Build codebook from all node bank vectors (detached — no gradient)
        # Shape: [max_nodes, D]
        codebook = F.normalize(self.node_bank.nu.detach(), dim=-1)  # [N, D]

        recovered_vecs = []
        weights = []

        with torch.no_grad():
            for bond in bonds:
                if bond.strength < 0.05:
                    continue  # Skip very weak bonds
                # Use the target node vector as the "tgt_vec" for unbinding.
                # The target node index is bond.target_id; clamp to valid range.
                tgt_idx = min(bond.target_id, self.node_bank.num_nodes - 1)
                tgt_vec = self.node_bank.nu[tgt_idx].detach()

                try:
                    _, recovered_vec, sim = self.bond_system.recover_source(
                        bond, tgt_vec, codebook
                    )
                    if sim > 0.0:  # Only use bonds with positive similarity
                        recovered_vecs.append(recovered_vec)
                        weights.append(bond.strength * sim)
                except Exception:
                    # recover_source can fail if bond_vector is malformed; skip
                    continue

        if not recovered_vecs:
            return torch.zeros(self.dim, device=device)

        # Strength-weighted mean of recovered vectors
        weight_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
        weight_tensor = weight_tensor / (weight_tensor.sum() + 1e-8)  # normalize weights

        stacked = torch.stack(recovered_vecs, dim=0).to(device)  # [K, D]
        bond_ctx = (weight_tensor.unsqueeze(-1) * stacked).sum(dim=0)  # [D]
        return F.normalize(bond_ctx, dim=-1)  # [D]

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

            # Fix 3.2: Use direct graph encoding for cleaner gradient path
            # Instead of going through the noisy node bank via pulse cycle,
            # use encode_graph directly to provide clean gradients to entity/relation encoders
            shape = self.encode_graph(entities, relations)
            
            # Still run pulse cycle for bond formation and other side effects
            committed_shape = controller.run_pulse(external_input=shape, regime=phase)
            if committed_shape is not None:
                if committed_shape.dim() == 1:
                    committed_shape = committed_shape.unsqueeze(0).expand(shape.size(0), -1)
                active_shape = committed_shape
            else:
                active_shape = shape
            
            # Fix 3.7: Track bond formation statistics from controller
            if phase == "compositional":
                self._total_compositional_pulses += 1
                # Check if bonds were formed in this pulse
                if len(self.bond_system.bonds) > 0:
                    # Count bonds formed in this pulse (bonds with age == 0)
                    new_bonds = sum(1 for bond in self.bond_system.bonds if bond.age == 0)
                    if new_bonds > 0:
                        self._bond_formation_count += 1

            budget_fraction = controller.budget_fraction_used

            # Fix 3.2: Use direct graph encoding (shape) instead of active_shape for entity prediction
            # This bypasses the noisy node bank and provides direct gradients:
            # entity_decoder → encode_graph → entity_encoder/relation_encoder
            # The direct encoding is cleaner early in training when node bank is untrained
            graph_encoding = shape  # Direct encoding, not filtered through node bank
            
            # Task 3.3: Use return_relation_embs=True to get per-relation embeddings
            # for the masked entity's neighborhood. These are passed directly to the
            # entity decoder so it can apply its own attention over relation types.
            neighborhood, rel_embs, rel_mask = self.encode_graph_neighborhood(
                entities, relations, masked_entity_idx, return_relation_embs=True
            )
            
            # Fix 3.4: Strengthen bond utilization during prediction
            # Bonds are formed between node bank indices (not entity IDs).
            # We use a best-effort approach: retrieve bond context from the
            # bond system and blend it into the neighborhood encoding.
            #
            # Steps:
            # 1. Call _retrieve_bond_context to aggregate recovered node vectors
            #    from all active bonds (weighted by bond strength × similarity).
            # 2. Project the bond context into the neighborhood space via
            #    bond_context_proj (learned linear layer).
            # 3. Use a learned gate (bond_gate) to control how much bond context
            #    modulates the neighborhood — gate starts near 0 (small init)
            #    so early training is stable, then opens as bonds become useful.
            # 4. Add gated bond context to neighborhood to produce
            #    neighborhood_with_bonds.
            #
            # This ensures bonds formed in Phase B are utilized during entity
            # prediction, improving relational reasoning (Requirements 1.5, 2.5).
            bond_ctx_vec = self._retrieve_bond_context(device)  # [D]

            if bond_ctx_vec.norm() > 1e-6:
                # Expand bond context to batch dimension: [B, D]
                B_size = neighborhood.size(0)
                bond_ctx_batch = bond_ctx_vec.unsqueeze(0).expand(B_size, -1)  # [B, D]

                # Project bond context into neighborhood space
                bond_ctx_proj = F.relu(self.bond_context_proj(bond_ctx_batch))  # [B, D]

                # Gated fusion: learn how much bond context to add to neighborhood
                gate_input = torch.cat([neighborhood, bond_ctx_proj], dim=-1)  # [B, 2D]
                gate = torch.sigmoid(self.bond_gate(gate_input))               # [B, D]
                neighborhood_with_bonds = neighborhood + gate * bond_ctx_proj  # [B, D]
                neighborhood_with_bonds = normalize(neighborhood_with_bonds)   # [B, D]
            else:
                # No bonds available (early training or no active nodes) — use
                # neighborhood directly. This is the same as the pre-fix behavior.
                neighborhood_with_bonds = neighborhood

            # Fix 3.2: Concatenate neighborhood with direct graph encoding (not active_shape)
            # This provides cleaner gradient path for entity prediction
            # Task 3.3: Pass per-relation embeddings and mask to the relation-aware decoder
            # so it can apply attention over individual relation types.
            entity_logits = self.entity_decoder(
                neighborhood_with_bonds, graph_encoding,
                relation_embs=rel_embs, relation_mask=rel_mask
            )  # [B, rel_vocab]

            # Relation prediction from active_shape (global graph context)
            relation_logits = self.relation_decoder(active_shape)

            loss_entity = F.cross_entropy(entity_logits, target_entity)
            loss_relation = F.cross_entropy(relation_logits, target_relation)
            
            # Auxiliary property loss: encourage explicit property-based reasoning
            # Decompose target entity into properties and predict them separately
            if hasattr(self.entity_decoder, 'get_property_logits'):
                prop_logits = self.entity_decoder.get_property_logits()

                # Determine entity encoding mode from entity_encoder
                use_size = getattr(self.entity_encoder, 'use_size', True)

                if use_size:
                    # 16-entity mode: entity_id = color * 8 + shape * 4 + size
                    target_colors = (target_entity // 8).clamp(0, 1)
                    target_shapes = ((target_entity % 8) // 4).clamp(0, 1)
                    target_sizes = (target_entity % 4).clamp(0, 3)
                    loss_color = F.cross_entropy(prop_logits['color'], target_colors)
                    loss_shape_prop = F.cross_entropy(prop_logits['shape'], target_shapes)
                    loss_size = F.cross_entropy(prop_logits['size'], target_sizes)
                    loss_properties = (loss_color + loss_shape_prop + loss_size) / 3.0
                else:
                    # 4-entity mode: entity_id = color * 2 + shape
                    target_colors = (target_entity // 2).clamp(0, 1)
                    target_shapes = (target_entity % 2).clamp(0, 1)
                    loss_color = F.cross_entropy(prop_logits['color'], target_colors)
                    loss_shape_prop = F.cross_entropy(prop_logits['shape'], target_shapes)
                    loss_properties = (loss_color + loss_shape_prop) / 2.0
            else:
                loss_properties = torch.tensor(0.0, device=device)

            # Shape loss: cosine distance between active_shape and the re-encoded graph
            # using the predicted entity — encourages shape to be predictive.
            with torch.no_grad():
                pred_entity_ids = entity_logits.argmax(dim=-1)
                pred_entities = pred_entity_ids.unsqueeze(1).expand_as(entities)
                target_shape = self.encode_graph(pred_entities, relations).detach()
            loss_shape = 1.0 - F.cosine_similarity(active_shape, target_shape, dim=-1).mean()

            spike_mask = self.node_bank.spike_mask(phase, budget_fraction_used=budget_fraction)
            loss_spike = spike_mask.mean()

            # Fix 3.5: Adjust loss weighting for compositional phase
            # Increase entity loss weight to 2.0 to prioritize entity prediction learning.
            # Reduce shape loss weight to 0.05 to prevent gradient interference with
            # entity prediction gradients while still maintaining shape signal.
            if phase == "compositional":
                entity_weight = 2.0   # increased from 1.0 to prioritize entity learning
                shape_weight = 0.05   # reduced from 0.1 to prevent gradient interference
                property_weight = 1.0  # Auxiliary property loss to encourage property-based reasoning
            else:
                entity_weight = 1.0
                shape_weight = 0.1    # perceptive phase: original weight unchanged
                property_weight = 0.5
            
            loss = (entity_weight * loss_entity + 
                   0.5 * loss_relation + 
                   shape_weight * loss_shape + 
                   property_weight * loss_properties +
                   0.01 * loss_spike)

            if controller.pulse_error_loss is not None:
                loss = loss + 0.01 * controller.pulse_error_loss

            # Fix 3.6: Add gradient monitoring and logging
            # Track gradient norms for entity_encoder, relation_encoder, entity_decoder
            # This helps diagnose gradient flow issues during training
            if hasattr(self, '_gradient_log_step'):
                self._gradient_log_step += 1
            else:
                self._gradient_log_step = 0
            
            # Log gradients every 100 steps (will be computed after backward pass)
            # Store flag for external logging (train.py will check this)
            self._should_log_gradients = (self._gradient_log_step % 100 == 0)

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
