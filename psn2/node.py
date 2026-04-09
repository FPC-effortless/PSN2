"""T02: NodeBank — stateful node population with nu, e, tau, sigma, budget.

Fixes applied:
  #1  update_errors() now returns a differentiable error tensor so gradients
      flow through the pulse cycle into nu.
  #7  silent_decay() is called from PhaseController after each pulse on
      non-spiking nodes.

PRD constants:
  lambda_tau        = 0.90
  lambda_e_silent   = 0.95
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LAMBDA_TAU = 0.90
LAMBDA_E_SILENT = 0.95


class NodeBank(nn.Module):
    NODE_TYPES = {"perceptive": 0, "compositional": 1, "recursive": 2, "commit": 3}

    def __init__(self, num_nodes: int, dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.nu = nn.Parameter(torch.randn(num_nodes, dim) * 0.02)
        self.register_buffer("e",         torch.zeros(num_nodes))
        self.register_buffer("tau",       torch.zeros(num_nodes))
        self.register_buffer("sigma",     torch.zeros(num_nodes, dtype=torch.long))
        self.register_buffer("b",         torch.ones(num_nodes))
        self.register_buffer("active",    torch.ones(num_nodes))
        self.register_buffer("committed", torch.zeros(num_nodes, dtype=torch.bool))

        # PRD Section 4.1 — extended state vector fields
        # pi_i: geometry bundle (metric128 + topo32 + dir32 + curvature16 + freq16 + occupancy32 + linguistic64 = 320)
        self.register_buffer("pi",          torch.zeros(num_nodes, 320))
        # kappa_i: bond capacity accumulator R^32
        self.register_buffer("kappa",       torch.zeros(num_nodes, 32))
        # mu_i: modality affinity R^6
        self.register_buffer("mu",          torch.zeros(num_nodes, 6))
        # chi_i: gate state R^8
        self.register_buffer("chi",         torch.zeros(num_nodes, 8))
        # omega_i: plasticity R^4
        self.register_buffer("omega",       torch.zeros(num_nodes, 4))
        # g_i: growth state R^4 — [spawn_pressure, prune_score, merge_candidate, age]
        self.register_buffer("growth_state", torch.zeros(num_nodes, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Cosine-similarity attention. x: [B,D] or [D] -> [B,N]."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return torch.matmul(x, self.nu.t())

    # ------------------------------------------------------------------
    # Fix #1: differentiable error computation
    # ------------------------------------------------------------------
    def compute_errors_differentiable(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-node L2 prediction error as a differentiable tensor.
        x: [B,D] or [D]
        Gradients flow back into self.nu through this path.

        Note (#17): multiplying by self.active (a float buffer, not a parameter)
        correctly zeros inactive nodes without affecting gradient flow to active
        nodes — self.active is not in the autograd graph, so it acts as a
        static mask. This is intentional: inactive nodes contribute zero to
        pulse_error_loss, which is correct behavior.
        """
        if x.dim() > 1:
            x_mean = x.mean(dim=0)          # [D]
        else:
            x_mean = x                       # [D]
        # [N,D] - [D] -> [N,D], then L2 norm -> [N]
        err = torch.norm(self.nu - x_mean.unsqueeze(0), dim=-1)  # differentiable
        return err * self.active

    def update_errors(self, x: torch.Tensor):
        """
        Update the non-differentiable e/tau buffers from input x.
        Called inside torch.no_grad() contexts (e.g. growth checks).
        """
        with torch.no_grad():
            err = self.compute_errors_differentiable(x).detach()
            self.e.copy_(err)
            self.tau.mul_(LAMBDA_TAU).add_(0.10 * err)

    # ------------------------------------------------------------------
    # Fix #7: silent_decay actually called (see PhaseController)
    # ------------------------------------------------------------------
    def silent_decay(self, spike_mask: torch.Tensor):
        """
        Decay e and tau for nodes that did NOT spike this pulse.
        spike_mask: [N] float, 1 = spiked, 0 = silent.
        """
        with torch.no_grad():
            silent = (spike_mask < 0.5) & self.active.bool()
            self.tau[silent] = self.tau[silent] * LAMBDA_TAU
            self.e[silent]   = self.e[silent]   * LAMBDA_E_SILENT

    def spike_mask(self, phase: str, thresholds: dict = None,
                   budget_fraction_used: float = 0.0) -> torch.Tensor:
        """Binary spike mask for given phase."""
        if thresholds is None:
            from psn2.sce import SpikeGate
            tau = SpikeGate().threshold(phase, budget_fraction_used)
        else:
            tau = thresholds.get(phase, 0.30)
        return (self.e > tau).float() * self.active

    def increment_age(self):
        """Increment g_i[:,3] (age) for all active nodes each pulse."""
        with torch.no_grad():
            self.growth_state[:, 3] += self.active

    def mark_committed(self, node_ids: torch.Tensor):
        with torch.no_grad():
            self.committed[node_ids] = True

    def reset_committed(self):
        with torch.no_grad():
            self.committed.fill_(False)
