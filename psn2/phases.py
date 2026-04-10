"""T14: Phase A-F Pulse Cycle.

Dual-mode design (addresses I-3 vs backprop critique):
  INFERENCE mode: local W_ff/W_fb update in Phase B runs continuously,
    no global backward pass. This is the biologically-plausible path.
  TRAINING mode: global AdamW backward pass runs AFTER the pulse cycle
    via pulse_error_loss accumulated in Phase A. The two modes are
    explicitly separated — Phase B local updates are no_grad and do not
    interact with the autograd graph. The global backward pass updates
    nu through the differentiable error path in Phase A only.

SCE sparsity note (addresses SCE/autograd critique):
  The spike gate is a FORWARD-PASS optimization only during training.
  The surrogate gradient exists for inference-time adaptation (I-13),
  not for the main training backward pass. During training, all nodes
  participate in the autograd graph via compute_errors_differentiable;
  the spike mask gates which nodes contribute to the pulse cycle output
  but does not create true sparse computation in the backward pass.
  True sparse inference (80-90% silent nodes) is realized at eval time
  when torch.no_grad() is active.

Other fixes:
  #1  pulse_error_loss accumulated in Phase A for global backward pass.
  #3  Verifier uses accumulated tau across pulses (persistent controller).
  #4  budget_fraction_used computed from actual pulse consumption.
  #7  silent_decay() called after every pulse on non-spiking nodes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional

from psn2.node import NodeBank
from psn2.vsa import normalize, bundle


class PhaseController:
    """
    Manages the A-F pulse cycle for one episode.
    Instantiate once per episode (not per batch step) so tau accumulates.
    """

    def __init__(self, node_bank: NodeBank, budget: int = 10,
                 bond_system=None, ess=None):
        self.node_bank = node_bank
        self.max_budget = budget
        self.budget = budget
        self.phases = ["A", "B", "C", "D", "E", "F"]
        self.phase_idx = 0
        self.committed_shape: Optional[torch.Tensor] = None
        self.active_regime = "perceptive"
        self.bond_system = bond_system
        self.ess = ess

        self._current_input: Optional[torch.Tensor] = None
        self._modulatory_input: Optional[torch.Tensor] = None
        self._coalition_stability: float = 0.0
        self._verifier_passed: bool = True

        # Fix #1: accumulated differentiable error loss across pulses
        self.pulse_error_loss: Optional[torch.Tensor] = None
        self._pulses_run: int = 0

    # ------------------------------------------------------------------
    # Fix #4: real budget fraction
    # ------------------------------------------------------------------
    @property
    def budget_fraction_used(self) -> float:
        return self._pulses_run / max(self.max_budget, 1)

    @property
    def current_phase(self):
        return self.phases[self.phase_idx]

    def execute_phase(self, phase: str):
        """Execute a specific phase by name. Used for testing."""
        if phase == "A":
            self._phase_a(self._current_input, self._modulatory_input)
        elif phase == "B":
            self._phase_b()
        elif phase == "C":
            self._phase_c()
        elif phase == "D":
            self._phase_d()
        elif phase == "E":
            self._phase_e()
        elif phase == "F":
            self._phase_f()

    def step_phase(self):
        self.phase_idx = (self.phase_idx + 1) % len(self.phases)

    def reset_pulse(self):
        self.phase_idx = 0

    def run_pulse(self, external_input: torch.Tensor,
                  modulatory_input: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Execute one full A-F pulse. Returns committed_shape."""
        if self.budget <= 0:
            return self.committed_shape

        self._current_input = external_input
        self._modulatory_input = modulatory_input

        for _ in range(len(self.phases)):
            phase = self.current_phase
            if phase == "A":
                self._phase_a(external_input, modulatory_input)
            elif phase == "B":
                self._phase_b()
            elif phase == "C":
                self._phase_c()
            elif phase == "D":
                self._phase_d()
            elif phase == "E":
                self._phase_e()
            elif phase == "F":
                self._phase_f()
            self.step_phase()

        self.budget -= 1
        self._pulses_run += 1
        return self.committed_shape

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _phase_a(self, external_input: Optional[torch.Tensor],
                 modulatory_input: Optional[torch.Tensor]):
        """
        Evidence integration.
        Fix #1: compute differentiable error and accumulate into
        pulse_error_loss so gradients reach node.nu.
        """
        if external_input is not None:
            # Differentiable path — gradients flow into nu
            err_diff = self.node_bank.compute_errors_differentiable(external_input)
            pulse_loss = err_diff.mean()
            if self.pulse_error_loss is None:
                self.pulse_error_loss = pulse_loss
            else:
                self.pulse_error_loss = self.pulse_error_loss + pulse_loss

            # Also update non-differentiable buffers for spike decisions
            self.node_bank.update_errors(external_input)

        if modulatory_input is not None:
            # Modulatory: attenuated, does not override proximal
            mod = modulatory_input * 0.3
            with torch.no_grad():
                mod_mean = mod.mean(dim=0) if mod.dim() > 1 else mod
                mod_err = torch.norm(self.node_bank.nu - mod_mean.unsqueeze(0), dim=-1) * 0.1
                self.node_bank.e.add_(mod_err)

    def _phase_b(self):
        """Bond update via VSA bind mechanics + local W_ff/W_fb weight update (PRD Section 4.3)."""
        if self.bond_system is not None:
            self.bond_system.pulse_decay()

        # Local weight update: W_ff and W_fb via prediction-error rule
        # lr_ff = lr_fb = 1e-4 (PRD frozen constant)
        # Fix #8: skip local update for batched input (B>1) — batch mean loses
        # task-specific signal. Only apply for single-sample inference.
        LR_FF = 1e-4
        inp = self._current_input
        if inp is not None and inp.dim() == 1:
            # Single sample: apply local update
            with torch.no_grad():
                active_mask = self.node_bank.active.bool()
                if active_mask.any():
                    e = self.node_bank.e[active_mask].unsqueeze(1)
                    delta = -LR_FF * e * (self.node_bank.nu.data[active_mask] - inp.unsqueeze(0))
                    self.node_bank.nu.data[active_mask] += delta
        # For batched input (dim > 1): local update skipped — global AdamW handles it

    @property
    def _sym_score(self) -> float:
        """Symmetry score cos(W_ff, W_fb).
        In this simplified model W_ff ≈ W_fb ≈ nu (same weight matrix),
        so symmetry is perfect by construction."""
        return 1.0

    def _phase_c(self):
        """Morphogenic field — emotional shape induction."""
        if self.ess is not None and self.committed_shape is not None:
            self.ess.induce(self.committed_shape)
            self.ess.pulse_decay()

    def _phase_d(self):
        """
        Shape formation — coalition detection, stability computation.
        Fix #3: tau accumulates across pulses (not reset), so stability
        grows meaningfully as the episode progresses.
        """
        from psn2.sce import SpikeGate
        tau_thresh = SpikeGate().threshold(self.active_regime, self.budget_fraction_used)
        coalition_mask = self.node_bank.tau > tau_thresh

        if coalition_mask.any():
            coalition_nodes = self.node_bank.nu[coalition_mask]
            if coalition_nodes.size(0) > 1:
                norm = F.normalize(coalition_nodes.detach(), dim=-1)
                sim_matrix = torch.mm(norm, norm.t())
                n = coalition_nodes.size(0)
                off_diag = sim_matrix[~torch.eye(n, dtype=torch.bool,
                                                  device=norm.device)]
                self._coalition_stability = float(off_diag.mean().item())
            else:
                self._coalition_stability = 1.0
        else:
            self._coalition_stability = 0.0

    def _phase_e(self):
        """
        Verifier gate.
        Fix #3: threshold scales with pulses run — early pulses are lenient,
        later pulses require genuine stability.
        """
        # Leniency: starts at 0.1, rises to 0.5 as budget is consumed
        required = 0.1 + 0.4 * self.budget_fraction_used
        self._verifier_passed = self._coalition_stability >= required

    def _phase_f(self):
        """
        Commitment — render buffer, dissolve unstable shapes.
        Fix #7: call silent_decay on non-spiking nodes.
        """
        from psn2.sce import SpikeGate
        tau_thresh = SpikeGate().threshold(self.active_regime, self.budget_fraction_used)
        spike_mask = (self.node_bank.e > tau_thresh).float() * self.node_bank.active

        # Fix #7: decay silent nodes
        self.node_bank.silent_decay(spike_mask)

        if not self._verifier_passed:
            return

        # Commit from stable nodes (tau > 0.85)
        stable_mask = self.node_bank.tau > 0.85
        if stable_mask.any():
            stable_nodes = self.node_bank.nu[stable_mask]
            self.committed_shape = normalize(stable_nodes.mean(dim=0).detach())
        elif self._current_input is not None:
            inp = self._current_input
            if inp.dim() > 1:
                inp = inp.mean(dim=0)
            self.committed_shape = normalize(inp.detach())
