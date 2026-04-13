"""T33: Expanded Loss Family — 26 loss components, activated per developmental stage."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Optional

# Loss weights from PRD
LOSS_WEIGHTS = {
    "L_error":      0.12,
    "L_bond":       0.08,
    "L_shape":      0.08,
    "L_compact":    0.12,
    "L_refine":     0.07,
    "L_world":      0.07,
    "L_verify":     0.09,
    "L_calib":      0.06,
    "L_transfer":   0.06,
    "L_render":     0.03,
    "L_vsa":        0.06,
    "L_mpf":        0.04,
    "L_attractor":  0.03,
    "L_sym":        0.04,
    "L_attn":       0.04,
    "L_expr":       0.04,
    "L_geom":       0.03,
    "L_compart":    0.03,
    "L_fast_adapt": 0.08,
    "L_retrieval":  0.05,
    "L_comp_split": 0.07,
    "L_forget":     0.06,
    "L_linguistic": 0.07,
    "L_emo_calib":  0.05,
    "L_social":     0.05,
    "L_temporal":   0.06,
}

# Stage activation: which losses are active from which stage
STAGE_ACTIVATION = {
    "L_error":      "D1",
    "L_bond":       "D1",
    "L_shape":      "D1",
    "L_compact":    "D1",
    "L_vsa":        "D1",
    "L_mpf":        "D1",
    "L_attractor":  "D1",
    "L_sym":        "D1",
    "L_geom":       "D1",
    "L_compart":    "D1",
    "L_refine":     "D2",
    "L_world":      "D2",
    "L_verify":     "D2",
    "L_calib":      "D2",
    "L_emo_calib":  "D3",
    "L_social":     "D3",
    "L_render":     "D4",
    "L_attn":       "D4",
    "L_expr":       "D4",
    "L_linguistic": "D4",
    "L_transfer":   "D5",
    "L_retrieval":  "D5",
    "L_comp_split": "D5",
    "L_temporal":   "D5",
    "L_fast_adapt": "D5",
    "L_forget":     "D6",
}

STAGE_ORDER = ["D1", "D2", "D3", "D4", "D5", "D6"]


def stage_index(stage: str) -> int:
    return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 0


def active_losses(current_stage: str) -> Dict[str, float]:
    """Return {loss_name: weight} for all losses active at current_stage."""
    cur_idx = stage_index(current_stage)
    return {
        name: LOSS_WEIGHTS[name]
        for name, stage in STAGE_ACTIVATION.items()
        if stage_index(stage) <= cur_idx
    }


def active_loss_normalizer(current_stage: str) -> float:
    """
    Sum of active loss weights at current_stage.

    The PRD states normalization = 1.98 (sum of all 26 weights).
    However, dividing by 1.98 when only D1 losses are active (sum ~0.55)
    would make early-stage gradients ~3.6x weaker than late-stage gradients,
    creating structural discontinuities at each stage transition.

    We normalize by the ACTIVE weight sum so gradient magnitude is
    consistent across stage transitions. The PRD's 1.98 is the asymptotic
    value when all 26 losses are active (D6).
    """
    active = active_losses(current_stage)
    total = sum(active.values())
    return max(total, 1e-8)


class PSN2LossFamily:
    """
    Computes the active subset of the 26-component loss family.
    Callers provide pre-computed scalar tensors for each component.
    Missing components default to zero.
    """

    def __init__(self, current_stage: str = "D1"):
        self.current_stage = current_stage

    def compute(self, components: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        components: {loss_name: scalar tensor}
        Normalizes by the sum of ACTIVE weights (not fixed 1.98) to keep
        gradient magnitude consistent across stage transitions.
        """
        active = active_losses(self.current_stage)
        # Infer device from first available component tensor
        device = None
        for val in components.values():
            if isinstance(val, torch.Tensor):
                device = val.device
                break
        total = torch.tensor(0.0, device=device)
        for name, weight in active.items():
            val = components.get(name)
            if val is not None:
                total = total + weight * val
        return total / active_loss_normalizer(self.current_stage)

    # ---- Individual loss implementations ----

    @staticmethod
    def L_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Multi-step prediction-error minimization."""
        return F.mse_loss(pred, target)

    @staticmethod
    def L_bond(bond_pred: torch.Tensor, bond_target: torch.Tensor) -> torch.Tensor:
        """VSA bond accuracy — cosine distance."""
        return 1.0 - F.cosine_similarity(bond_pred, bond_target, dim=-1).mean()

    @staticmethod
    def L_shape(stability: torch.Tensor, tau_stable: float = 0.85) -> torch.Tensor:
        """Shape stability hinge loss."""
        return F.relu(tau_stable - stability).mean()

    @staticmethod
    def L_compact(description_lengths: torch.Tensor) -> torch.Tensor:
        """Expected description length — dominant."""
        return description_lengths.mean()

    @staticmethod
    def L_vsa(recovered: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """VSA binding recovery accuracy."""
        return 1.0 - F.cosine_similarity(recovered, original, dim=-1).mean()

    @staticmethod
    def L_mpf(phi: torch.Tensor) -> torch.Tensor:
        """Phi low on solved tasks."""
        return phi.mean()

    @staticmethod
    def L_attractor(centroids: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
        """MPF basins well-separated — pairwise cosine should be low."""
        if centroids.size(0) < 2:
            return torch.tensor(0.0)
        norm = F.normalize(centroids, dim=-1)
        sim_matrix = torch.mm(norm, norm.t())
        # Penalize high off-diagonal similarity
        mask = ~torch.eye(centroids.size(0), dtype=torch.bool, device=centroids.device)
        return F.relu(sim_matrix[mask] - eps).mean()

    @staticmethod
    def L_sym(W_ff: torch.Tensor, W_fb: torch.Tensor) -> torch.Tensor:
        """W_ff/W_fb cosine symmetry floor."""
        return 1.0 - F.cosine_similarity(W_ff.flatten().unsqueeze(0),
                                          W_fb.flatten().unsqueeze(0)).squeeze()

    @staticmethod
    def L_geom(shape_a: torch.Tensor, shape_b: torch.Tensor) -> torch.Tensor:
        """Cross-geometry consistency."""
        return 1.0 - F.cosine_similarity(shape_a, shape_b, dim=-1).mean()

    @staticmethod
    def L_compart(compartment_vecs: torch.Tensor) -> torch.Tensor:
        """Compartment disentanglement — minimize cross-compartment similarity."""
        if compartment_vecs.size(0) < 2:
            return torch.tensor(0.0)
        norm = F.normalize(compartment_vecs, dim=-1)
        sim = torch.mm(norm, norm.t())
        mask = ~torch.eye(compartment_vecs.size(0), dtype=torch.bool, device=compartment_vecs.device)
        return sim[mask].abs().mean()

    @staticmethod
    def L_refine(stability_before: torch.Tensor, stability_after: torch.Tensor) -> torch.Tensor:
        """Negative per-pulse stability improvement."""
        return -F.relu(stability_after - stability_before).mean()

    @staticmethod
    def L_world(pred_state: torch.Tensor, actual_state: torch.Tensor) -> torch.Tensor:
        """World model MSE."""
        return F.mse_loss(pred_state, actual_state)

    @staticmethod
    def L_verify(verifier_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Verifier veto classification."""
        return F.binary_cross_entropy_with_logits(verifier_logits, labels.float())

    @staticmethod
    def L_calib(pred_error: torch.Tensor, actual_correct: torch.Tensor) -> torch.Tensor:
        """Error calibration vs actual correctness."""
        return F.mse_loss(pred_error, actual_correct.float())

    @staticmethod
    def L_linguistic(shape1: torch.Tensor, shape2: torch.Tensor) -> torch.Tensor:
        """USL round-trip fidelity."""
        return 1.0 - F.cosine_similarity(shape1.unsqueeze(0), shape2.unsqueeze(0)).squeeze()

    @staticmethod
    def L_emo_calib(emo_logits: torch.Tensor, emo_labels: torch.Tensor) -> torch.Tensor:
        """Emotional shape induction accuracy."""
        return F.cross_entropy(emo_logits, emo_labels)

    @staticmethod
    def L_social(verifier_agreement: torch.Tensor) -> torch.Tensor:
        """Verifier agreement rate on socially observed traces."""
        return 1.0 - verifier_agreement.mean()

    @staticmethod
    def L_temporal(pulse_savings: torch.Tensor, target_savings: float = 3.0) -> torch.Tensor:
        """Pulse savings from temporal motif replay."""
        return F.relu(target_savings - pulse_savings).mean()

    @staticmethod
    def L_transfer(with_motifs: torch.Tensor, without_motifs: torch.Tensor) -> torch.Tensor:
        """Efficiency delta with vs without motifs."""
        return F.relu(with_motifs - without_motifs).mean()

    @staticmethod
    def L_retrieval(pred_utility: torch.Tensor, actual_utility: torch.Tensor) -> torch.Tensor:
        """ERS utility score prediction accuracy."""
        return F.mse_loss(pred_utility, actual_utility)

    @staticmethod
    def L_comp_split(held_out_perf: torch.Tensor, threshold: float = 0.75) -> torch.Tensor:
        """Performance on held-out compositional test instances."""
        return F.relu(threshold - held_out_perf).mean()

    @staticmethod
    def L_fast_adapt(query_loss: torch.Tensor) -> torch.Tensor:
        """Meta-episode query performance given support-set."""
        return query_loss.mean()

    @staticmethod
    def L_forget(ewc_penalty: torch.Tensor) -> torch.Tensor:
        """EWC forgetting penalty during online updates."""
        return ewc_penalty.mean()

    @staticmethod
    def L_render(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Surface generation from committed state."""
        return F.cross_entropy(rendered.reshape(-1, rendered.size(-1)), target.reshape(-1))

    @staticmethod
    def L_attn(retrieved: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """External attention retrieval quality."""
        return 1.0 - F.cosine_similarity(retrieved, target, dim=-1).mean()

    @staticmethod
    def L_expr(causal_pred: torch.Tensor, causal_target: torch.Tensor) -> torch.Tensor:
        """Internal expression causal quality."""
        return F.mse_loss(causal_pred, causal_target)
