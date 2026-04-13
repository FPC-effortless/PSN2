"""Spiking Computation Engine (SCE) — spike decision with surrogate gradient.

PRD constants:
  tau_spike_perceptive    = 0.30 + 0.10 * budget_fraction_used
  tau_spike_compositional = 0.20 + 0.05 * budget_fraction_used
  tau_spike_recursive     = 0.10
  tau_spike_commit        = 0.40
  Surrogate: 1 / (1 + |e_i - tau_spike|)^2

Sparsity / autograd note (addresses SCE/backprop structural incompatibility):
  During TRAINING (torch.is_grad_enabled()):
    - The spike mask gates the pulse cycle output in the forward pass.
    - The surrogate gradient allows gradients to flow through the hard
      threshold for inference-time adaptation (I-13 fast stream).
    - The main training backward pass flows through
      compute_errors_differentiable() in Phase A, NOT through the spike
      mask. All nodes are in the autograd graph during training.
    - True sparse computation (80-90% silent nodes) is a FORWARD-PASS
      optimization only — it does not reduce backward-pass memory.
  During INFERENCE (torch.no_grad()):
    - Silent nodes (spike=0) skip all Phase A-F computation entirely.
    - This is where the energy efficiency claim is realized.
    - Energy scales with task novelty as specified in PRD Section 5.3.

  This is an explicit design choice: training uses dense backprop for
  stability; inference uses sparse event-driven computation for efficiency.
  The two modes are not in conflict — they serve different objectives.
"""
from __future__ import annotations

import torch


class SpikeGate(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def threshold(self, phase: str, budget_fraction_used: float = 0.0) -> float:
        if phase == "perceptive":
            return 0.30 + 0.10 * budget_fraction_used
        elif phase == "compositional":
            return 0.20 + 0.05 * budget_fraction_used
        elif phase == "recursive":
            return 0.10
        elif phase == "commit":
            return 0.40
        return 0.30

    def forward(self, error: torch.Tensor, phase: str,
                budget_fraction_used: float = 0.0) -> torch.Tensor:
        """
        Returns spike gate tensor.

        During training: straight-through surrogate gradient for I-13 fast
        adaptation path. Main training gradients flow through Phase A error,
        not through this gate.

        During inference (no_grad): hard binary mask — silent nodes (0)
        consume no compute in subsequent phases.

        Surrogate: 1 / (1 + |e_i - tau|)^2
        """
        tau = self.threshold(phase, budget_fraction_used)
        hard = (error > tau).float()

        if not torch.is_grad_enabled():
            # Pure inference: return hard binary mask, no surrogate needed
            return hard

        # Training: straight-through estimator for I-13 fast adaptation
        soft = 1.0 / (1.0 + (error - tau).abs() ** 2)
        return hard + soft - soft.detach()

    def sparsity(self, error: torch.Tensor, phase: str,
                 budget_fraction_used: float = 0.0) -> float:
        """Fraction of silent nodes. Meaningful only during inference."""
        with torch.no_grad():
            tau = self.threshold(phase, budget_fraction_used)
            return float((error <= tau).float().mean().item())
