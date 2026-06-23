"""BehaviorCloning — supervised losses against human / recovery demonstrations.

Demonstration transitions are expected to carry control-mode metadata in the
replay buffer:
    info['control_mode'] = 'human' | 'autonomous'
    info['demo_type']    = 'normal' | 'failure_recovery'
This module is loss-only; the trainer is responsible for sampling and
combining BC + recovery-BC losses with the Dreamer actor loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class BehaviorCloning:
    def __init__(self, loss_type: str = "mse"):
        if loss_type not in ("mse", "l1"):
            raise ValueError(
                f"BehaviorCloning: loss_type must be 'mse' or 'l1', got '{loss_type}'")
        self.loss_type = loss_type

    # ──────────────────────────────────────────────────────────────────────
    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"BC pred/target shape mismatch: {tuple(pred.shape)} vs "
                f"{tuple(target.shape)}")
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        return F.l1_loss(pred, target)

    # ──────────────────────────────────────────────────────────────────────
    def compute_bc_loss(
        self,
        predicted_action: torch.Tensor,
        human_action: torch.Tensor,
    ) -> torch.Tensor:
        """Standard human-demonstration imitation loss."""
        return self._loss(predicted_action, human_action)

    def compute_recovery_bc_loss(
        self,
        predicted_action: torch.Tensor,
        recovery_action: torch.Tensor,
    ) -> torch.Tensor:
        """Imitation loss against actions recorded during human recovery
        from a previously-failed mission. Same form as `compute_bc_loss`
        — kept as a separate entry point so the trainer can weight it
        independently."""
        return self._loss(predicted_action, recovery_action)
