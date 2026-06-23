"""SafetyShield — clamps the policy's action by BEV obstacle risk + IMU tilt.

Used at inference time, between the policy and the actuator. Operates on
torch tensors so it can run either on GPU during rollout or on CPU in a
ROS 2 wrapper. No ROS / gz dependencies.

Conventions:
    action     : tensor [..., 2] = (linear_x, angular_z) in valid range
    obs['bev'] : tensor [B, 4, H, W] with channel 0 = occupancy in [0, 1]
    obs['imu'] : tensor [B, 12]; indices 6 = roll, 7 = pitch (radians)
"""
from __future__ import annotations

from typing import Dict, Tuple

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
    def estimate_obstacle_risk(self, lidar_bev: torch.Tensor) -> torch.Tensor:
        """Fraction of forward-cone cells occupied (BEV channel 0).

        Forward cone: rows 0..H/2 (BEV centred on rover, +X = up = "forward"
        per the obs_builder convention).
        """
        if lidar_bev.dim() != 4:
            raise ValueError(
                f"estimate_obstacle_risk expects [B, 4, H, W], got "
                f"{tuple(lidar_bev.shape)}")
        B, _, H, W = lidar_bev.shape
        occ = lidar_bev[:, 0]                                    # [B, H, W]
        forward = occ[:, : H // 2, W // 4 : 3 * W // 4]          # forward arc
        return forward.mean(dim=(-1, -2)).clamp(0.0, 1.0)        # [B]

    def estimate_tilt_risk(self, imu: torch.Tensor) -> torch.Tensor:
        """tilt = hypot(roll, pitch)  [B]   (radians)"""
        if imu.dim() != 2 or imu.shape[1] < 8:
            raise ValueError(
                f"estimate_tilt_risk expects [B, >=8], got {tuple(imu.shape)}")
        roll = imu[:, 6]
        pitch = imu[:, 7]
        return torch.hypot(roll, pitch)

    # ──────────────────────────────────────────────────────────────────────
    def clip_action(self, action: torch.Tensor) -> torch.Tensor:
        low = self.action_low.to(action.device)
        high = self.action_high.to(action.device)
        lin = action[..., 0].clamp(min=low[0], max=high[0])
        ang = action[..., 1].clamp(min=low[1], max=high[1])
        return torch.stack([lin, ang], dim=-1)

    # ──────────────────────────────────────────────────────────────────────
    def filter_action(
        self,
        action: torch.Tensor,
        obs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply BEV+IMU+limits filtering, return safe action."""
        a = action.clone()
        if a.dim() == 1:
            a = a.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        lin, ang = a[..., 0], a[..., 1]

        # Obstacle risk from BEV.
        # Accept either "lidar_bev" (key used by the ROS driver) or "bev"
        # (older key) so the shield works regardless of obs naming.
        bev_tensor = obs.get("lidar_bev", obs.get("bev"))
        if bev_tensor is not None:
            risk = self.estimate_obstacle_risk(bev_tensor)
            # Smooth brake between [risk_threshold, stop_threshold]
            risk_t = self.cfg.obstacle_risk_threshold
            stop_t = self.cfg.obstacle_stop_threshold
            denom = max(stop_t - risk_t, 1e-3)
            brake = ((stop_t - risk) / denom).clamp(min=self.cfg.brake_linear_scale_min,
                                                    max=1.0)
            lin = lin * brake
            # Above stop_threshold: hard stop + emergency turn.
            emergency = risk > stop_t
            lin = torch.where(emergency, torch.zeros_like(lin), lin)
            kick = torch.full_like(ang, self.cfg.avoid_angular_kick)
            ang = torch.where(emergency & (ang.abs() < 1e-3), kick, ang)

        # Tilt risk from IMU.
        if "imu" in obs:
            tilt = self.estimate_tilt_risk(obs["imu"])
            brake_t = self.cfg.tilt_brake_rad
            stop_t = self.cfg.tilt_stop_rad
            denom = max(stop_t - brake_t, 1e-3)
            tilt_brake = ((stop_t - tilt) / denom).clamp(
                min=self.cfg.brake_linear_scale_min, max=1.0)
            lin = lin * tilt_brake
            tilt_emergency = tilt > stop_t
            lin = torch.where(tilt_emergency, torch.zeros_like(lin), lin)

        out = torch.stack([lin, ang], dim=-1)
        out = self.clip_action(out)
        return out.squeeze(0) if squeeze else out
