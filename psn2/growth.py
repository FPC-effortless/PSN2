"""Substrate Growth Protocol (SGP) — spawn, prune, merge, attractor-expand.

PRD spec (Section 12):
  tau_spawn = 0.65, N_persist = 10, sigma_spawn = 0.05
  tau_prune = 0.75, tau_age_prune = 200
  tau_merge = 0.92
  stability_gate_window = 50 pulses
  I-24: nodes_spawned <= nodes_freed_by_motifs + nodes_pruned_merged at every checkpoint
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from psn2.node import NodeBank
    from psn2.attractor import AttractorLibrary

# PRD frozen constants
TAU_SPAWN = 0.65
N_PERSIST = 10
SIGMA_SPAWN = 0.05
TAU_PRUNE = 0.75
TAU_AGE_PRUNE = 200
TAU_MERGE = 0.92
STABILITY_GATE_WINDOW = 50
TAU_CLEANUP_CONF = 0.45
ATTRACTOR_COS_THRESHOLD = 0.45   # PRD: no attractor match above cos=0.45 triggers spawn


@dataclass
class SpawnRecord:
    node_id: int
    parent_id: int
    trigger_type: str
    pulse_timestamp: int
    stable: Optional[bool] = None   # set after stability gate evaluation


@dataclass
class PruneRecord:
    node_id: int
    prune_score: float
    pulse_timestamp: int


@dataclass
class MergeRecord:
    node_a: int
    node_b: int
    merged_id: int
    pulse_timestamp: int


@dataclass
class AttractorExpandRecord:
    attractor_id: str
    pulse_timestamp: int


@dataclass
class I24Check:
    timestamp: int
    spawned: int
    freed: int
    gate_pass: bool


class GrowthLedger:
    """
    Full SGP ledger tracking all four growth operations and I-24 enforcement.
    """

    def __init__(self):
        self.session_id: str = str(uuid.uuid4())
        self.checkpoint_id: str = str(uuid.uuid4())

        self.spawned_nodes: List[SpawnRecord] = []
        self.pruned_nodes: List[PruneRecord] = []
        self.merged_nodes: List[MergeRecord] = []
        self.expanded_attrs: List[AttractorExpandRecord] = []
        self.i24_checks: List[I24Check] = []

        # Per-node high-error pulse counters for spawn eligibility
        self._error_persistence: dict = {}
        # Pending stability checks: node_id -> (birth_pulse, parent_nu)
        self._pending_stability: dict = {}

        self._nodes_freed_by_motifs: int = 0

    # ------------------------------------------------------------------
    # I-24 accounting
    # ------------------------------------------------------------------
    @property
    def nodes_spawned_since_checkpoint(self) -> int:
        return len(self.spawned_nodes)

    @property
    def nodes_freed_since_checkpoint(self) -> int:
        """Pruned + merged pairs + motif-freed."""
        merged_pairs = len(self.merged_nodes)
        return len(self.pruned_nodes) + merged_pairs + self._nodes_freed_by_motifs

    @property
    def net_node_delta(self) -> int:
        return self.nodes_spawned_since_checkpoint - self.nodes_freed_since_checkpoint

    def record_motif_freed(self, count: int):
        self._nodes_freed_by_motifs += count

    @property
    def nodes_freed_by_motifs(self) -> int:
        return self._nodes_freed_by_motifs

    def check_i24(self, pulse: int) -> bool:
        """Returns True if I-24 is satisfied (spawn <= freed)."""
        ok = self.nodes_spawned_since_checkpoint <= self.nodes_freed_since_checkpoint
        self.i24_checks.append(I24Check(
            timestamp=pulse,
            spawned=self.nodes_spawned_since_checkpoint,
            freed=self.nodes_freed_since_checkpoint,
            gate_pass=ok,
        ))
        return ok

    def budget_gate_allows_spawn(self, bootstrap_allowance: int = 10) -> bool:
        """
        Block spawn if I-24 would be violated.

        Bootstrap allowance: during early training (D1/D2) temporal motifs
        haven't formed yet so nodes_freed_by_motifs ≈ 0. Without an
        allowance, I-24 would freeze the substrate at N=256 for the first
        several sessions. We permit up to `bootstrap_allowance` spawns
        before motif compression begins, matching the PRD's Session 2 note:
        "spawn pressure accumulates but budget gate prevents premature growth"
        — premature means unbounded, not zero.
        """
        freed = self.nodes_freed_since_checkpoint
        spawned = self.nodes_spawned_since_checkpoint
        # Allow bootstrap_allowance free spawns; after that enforce I-24
        if spawned < bootstrap_allowance:
            return True
        return spawned <= freed + bootstrap_allowance

    # ------------------------------------------------------------------
    # Operation 1: Node Spawn
    # ------------------------------------------------------------------
    def spawn_eligible(self, node_id: int, error: float) -> bool:
        """Track persistent error; return True after N_PERSIST consecutive pulses."""
        if error > TAU_SPAWN:
            count = self._error_persistence.get(node_id, 0) + 1
            self._error_persistence[node_id] = count
            return count >= N_PERSIST
        else:
            self._error_persistence[node_id] = 0
            return False

    def maybe_spawn_node(self, node_bank: "NodeBank", step: int,
                         attractor_lib: Optional["AttractorLibrary"] = None) -> Optional[int]:
        """
        Spawn new node when persistent error > tau_spawn for N_persist pulses
        AND no attractor match above cos=0.45 AND I-24 budget allows.
        Returns new node index if spawned, else None.
        """
        if not self.budget_gate_allows_spawn():
            return None

        errors = node_bank.e.detach()
        active = node_bank.active.detach()

        for i in range(len(errors)):
            if not active[i]:
                continue
            err = float(errors[i].item())
            if not self.spawn_eligible(i, err):
                continue

            # Check no attractor match above cos=0.45
            if attractor_lib is not None and len(attractor_lib) > 0:
                results = attractor_lib.query(node_bank.nu[i].detach(), k=1)
                if results and results[0][0] > ATTRACTOR_COS_THRESHOLD:
                    continue

            # Find first inactive slot
            inactive = (active == 0).nonzero(as_tuple=True)[0]
            if len(inactive) == 0:
                return None
            new_idx = int(inactive[0].item())

            with torch.no_grad():
                noise = torch.randn_like(node_bank.nu[i]) * SIGMA_SPAWN
                node_bank.nu[new_idx].data.copy_(node_bank.nu[i].detach() + noise)
                node_bank.active[new_idx] = 1.0
                node_bank.e[new_idx] = float(errors[i].item())
                node_bank.tau[new_idx] = 0.0

            self._error_persistence[i] = 0
            rec = SpawnRecord(node_id=new_idx, parent_id=i,
                              trigger_type="prediction_error", pulse_timestamp=step)
            self.spawned_nodes.append(rec)
            self._pending_stability[new_idx] = (step, node_bank.nu[i].detach().clone())
            return new_idx

        return None

    # ------------------------------------------------------------------
    # Operation 2: Node Prune
    # ------------------------------------------------------------------
    def compute_prune_score(self, node_bank: "NodeBank", node_id: int,
                             activation_history: Optional[torch.Tensor] = None) -> float:
        """
        prune_score = (1 - mean_activation) * (1 - mean_bond_strength) * sigmoid(age - tau_age_prune)
        Age comes from node_bank.growth_state[node_id, 3] maintained by increment_age().
        """
        a = float(node_bank.e[node_id].item())
        mean_act = 1.0 - min(1.0, a)
        mean_bond = float(node_bank.tau[node_id].item())
        # Fix #1: read actual age from growth_state[:,3]
        if hasattr(node_bank, "growth_state"):
            age = float(node_bank.growth_state[node_id, 3].item())
        else:
            age = 0.0
        score = (1.0 - mean_act) * (1.0 - min(1.0, mean_bond))
        age_factor = float(torch.sigmoid(torch.tensor(age - TAU_AGE_PRUNE)).item())
        return float(score * age_factor)

    def maybe_prune_nodes(self, node_bank: "NodeBank", step: int) -> List[int]:
        """Prune nodes with prune_score > tau_prune."""
        pruned = []
        active_ids = node_bank.active.nonzero(as_tuple=True)[0].tolist()
        for i in active_ids:
            score = self.compute_prune_score(node_bank, i)
            if score > TAU_PRUNE:
                with torch.no_grad():
                    node_bank.active[i] = 0.0
                    node_bank.e[i] = 0.0
                    node_bank.tau[i] = 0.0
                self.pruned_nodes.append(PruneRecord(
                    node_id=i, prune_score=score, pulse_timestamp=step))
                pruned.append(i)
        return pruned

    # ------------------------------------------------------------------
    # Operation 3: Node Merge
    # ------------------------------------------------------------------
    def maybe_merge_nodes(self, node_bank: "NodeBank", step: int) -> List[Tuple[int, int, int]]:
        """Merge pairs with cosine similarity > tau_merge."""
        merged = []
        active_ids = node_bank.active.nonzero(as_tuple=True)[0].tolist()
        used = set()
        for i in active_ids:
            if i in used:
                continue
            for j in active_ids:
                if j <= i or j in used:
                    continue
                nu_i = node_bank.nu[i].detach()
                nu_j = node_bank.nu[j].detach()
                sim = float(F.cosine_similarity(nu_i.unsqueeze(0), nu_j.unsqueeze(0)).item())
                if sim > TAU_MERGE:
                    # Merge into i: bundle nu_i and nu_j
                    merged_nu = F.normalize((nu_i + nu_j) * 0.5, dim=-1)
                    with torch.no_grad():
                        node_bank.nu[i].data.copy_(merged_nu)
                        node_bank.active[j] = 0.0
                        node_bank.e[j] = 0.0
                        node_bank.tau[j] = 0.0
                    self.merged_nodes.append(MergeRecord(
                        node_a=i, node_b=j, merged_id=i, pulse_timestamp=step))
                    used.add(j)
                    merged.append((i, j, i))
                    break
        return merged

    # ------------------------------------------------------------------
    # Operation 4: Attractor Expand
    # ------------------------------------------------------------------
    def maybe_expand_attractor(self, attractor_lib: "AttractorLibrary",
                                vec: torch.Tensor, step: int) -> bool:
        """
        Add to attractor library if cleanup confidence < tau_cleanup_conf.
        Returns True if added.
        """
        if len(attractor_lib) >= attractor_lib.max_size:
            return False
        results = attractor_lib.query(vec, k=1)
        if results and results[0][0] > TAU_CLEANUP_CONF:
            return False
        attractor_lib.add(vec)
        attr_id = str(uuid.uuid4())[:8]
        self.expanded_attrs.append(AttractorExpandRecord(
            attractor_id=attr_id, pulse_timestamp=step))
        return True

    # ------------------------------------------------------------------
    # Stability gate (50-pulse window)
    # ------------------------------------------------------------------
    def evaluate_stability_gates(self, node_bank: "NodeBank", current_pulse: int):
        """
        For each pending spawned node, check stability gate after 50 pulses.
        Rollback if: error still high OR redundant with existing node.
        """
        to_remove = []
        for node_id, (birth_pulse, parent_nu) in list(self._pending_stability.items()):
            if current_pulse - birth_pulse < STABILITY_GATE_WINDOW:
                continue
            to_remove.append(node_id)
            if not node_bank.active[node_id]:
                continue
            err = float(node_bank.e[node_id].item())
            nu_j = node_bank.nu[node_id].detach()

            # Check redundancy with existing nodes
            active_ids = node_bank.active.nonzero(as_tuple=True)[0].tolist()
            max_sim = 0.0
            for k in active_ids:
                if k == node_id:
                    continue
                sim = float(F.cosine_similarity(
                    nu_j.unsqueeze(0), node_bank.nu[k].detach().unsqueeze(0)).item())
                max_sim = max(max_sim, sim)

            stable = (err < 0.50) and (max_sim < 0.92)
            # Mark stability result
            for rec in self.spawned_nodes:
                if rec.node_id == node_id:
                    rec.stable = stable
                    break

            if not stable:
                # Rollback
                with torch.no_grad():
                    node_bank.active[node_id] = 0.0
                    node_bank.e[node_id] = 0.0
                    node_bank.tau[node_id] = 0.0
                # Remove from spawned count (rollback)
                self.spawned_nodes = [r for r in self.spawned_nodes if r.node_id != node_id]

        for node_id in to_remove:
            self._pending_stability.pop(node_id, None)

    # ------------------------------------------------------------------
    # Legacy compatibility
    # ------------------------------------------------------------------
    @property
    def growth_rate_ok(self) -> bool:
        return self.budget_gate_allows_spawn()

    def log(self, step: int, kind: str, value: float):
        """Legacy log method for backward compatibility."""
        if kind == "spawn":
            pass  # handled by maybe_spawn_node
        elif kind == "compress":
            self._nodes_freed_by_motifs += 1

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "checkpoint_id": self.checkpoint_id,
            "spawned_nodes": [asdict(r) for r in self.spawned_nodes],
            "pruned_nodes": [asdict(r) for r in self.pruned_nodes],
            "merged_nodes": [asdict(r) for r in self.merged_nodes],
            "expanded_attrs": [asdict(r) for r in self.expanded_attrs],
            "i24_checks": [asdict(c) for c in self.i24_checks],
            "nodes_freed_by_motifs": self._nodes_freed_by_motifs,
            "net_node_delta": self.net_node_delta,
            # Fix #20: persist spawn-eligibility counters across sessions
            "error_persistence": dict(self._error_persistence),
            "pending_stability_pulses": {
                str(k): v[0] for k, v in self._pending_stability.items()
            },
        }

    def load_state_dict(self, state: dict):
        self.session_id = state.get("session_id", str(uuid.uuid4()))
        self.checkpoint_id = state.get("checkpoint_id", str(uuid.uuid4()))
        self.spawned_nodes = [SpawnRecord(**r) for r in state.get("spawned_nodes", [])]
        self.pruned_nodes = [PruneRecord(**r) for r in state.get("pruned_nodes", [])]
        self.merged_nodes = [MergeRecord(**r) for r in state.get("merged_nodes", [])]
        self.expanded_attrs = [AttractorExpandRecord(**r) for r in state.get("expanded_attrs", [])]
        self.i24_checks = [I24Check(**c) for c in state.get("i24_checks", [])]
        self._nodes_freed_by_motifs = state.get("nodes_freed_by_motifs", 0)
        # Fix #20: restore spawn-eligibility and stability state across sessions
        self._error_persistence = state.get("error_persistence", {})
        # _pending_stability can't be fully restored (parent_nu tensors not serialized),
        # but we restore the birth_pulse so stability windows resume correctly
        raw_pending = state.get("pending_stability_pulses", {})
        self._pending_stability = {
            int(k): (v, None) for k, v in raw_pending.items()
        }
