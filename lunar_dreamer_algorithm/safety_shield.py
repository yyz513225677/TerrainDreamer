"""Safety shield — clamps the policy's action by lidar proximity and limits.

Used at *inference* time, between the policy and the actuator. Operates on
raw torch tensors so it can be applied either on the GPU during rollout or
on the CPU in a ROS wrapper.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .config import SafetyConfig


class SafetyShield:
    def __init__(
        self,
        cfg: SafetyConfig,
        action_low: Tuple[float, float] = (0.0, -1.0),
        action_high: Tuple[float, float] = (0.8, 1.0),
    ):
        self.cfg = cfg
        self.action_low = torch.tensor(action_low)
        self.action_high = torch.tensor(action_high)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def min_distance(scan: torch.Tensor) -> torch.Tensor:
        """Min finite range in a 1-D laser scan vector. Inf/NaN are ignored.

        scan : [B, N] or [N]
        return : [B] or scalar
        """
        valid = torch.where(torch.isfinite(scan) & (scan > 0.05),
                            scan, torch.full_like(scan, float("inf")))
        return valid.min(dim=-1).values

    # ──────────────────────────────────────────────────────────────────────
    def apply(
        self,
        action: torch.Tensor,
        scan: Optional[torch.Tensor] = None,
        forward_min: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Shield a (batched) action tensor [..., 2] = (linear, angular).

        scan        : optional 1-D lidar slice, [B, N] or [N]
        forward_min : pre-computed forward-cone min range. If neither is
                      given the shield only enforces action limits.
        """
        a = action.clone()
        lin, ang = a[..., 0], a[..., 1]

        if forward_min is None and scan is not None:
            forward_min = self.min_distance(scan)

        if forward_min is not None:
            brake = self.cfg.lidar_brake_dist
            stop = self.cfg.lidar_stop_dist
            scale_floor = self.cfg.brake_linear_scale_min
            # Smooth linear scaling between stop_dist and brake_dist.
            brake_scale = ((forward_min - stop) / (brake - stop)).clamp(
                min=scale_floor, max=1.0)
            lin = lin * brake_scale
            # Hard stop + small avoidance kick when too close.
            too_close = forward_min < stop
            lin = torch.where(too_close, torch.zeros_like(lin), lin)
            kick = torch.where(
                too_close,
                torch.full_like(ang, self.cfg.avoid_angular_kick),
                torch.zeros_like(ang),
            )
            # Preserve sign of existing angular; add kick only if ang≈0.
            ang = torch.where(too_close & (ang.abs() < 1e-3), kick, ang)

        low = self.action_low.to(a.device)
        high = self.action_high.to(a.device)
        lin = lin.clamp(min=low[0], max=high[0])
        ang = ang.clamp(min=low[1], max=high[1])
        return torch.stack([lin, ang], dim=-1)
