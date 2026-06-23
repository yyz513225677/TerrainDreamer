"""FlagSeekingPolicy — hand-crafted base policy that always drives the rover
toward the current goal target.

Intended use:
  - Wrapper computes the goal_vector each step (as it already does for the
    learned actor).
  - Wrapper passes the base-policy action into ``model.select_action`` as the
    ``human_action`` argument, with ``control_mode=CONTROL_MODE_HUMAN``.
  - HumanTakeover forwards it to the cmd publisher AND tags the resulting
    transition as a human demo so the trainer's BC loss makes the learned
    actor imitate it.
  - When the actor is good enough, the wrapper can flip back to
    ``control_mode=CONTROL_MODE_AUTO`` to deploy the learned policy.

Pure-PyTorch; no ROS or sim imports.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class FlagSeekingPolicyConfig:
    v_max: float = 0.6                              # forward speed when aligned (reverted: 0.45 had worse mid_fail on extreme)
    omega_max: float = 0.8                          # max angular speed
    kp_omega: float = 1.2                           # P-gain on bearing
    align_cone_rad: float = math.radians(60.0)      # outside cone → turn in place
    stop_distance_m: float = 0.4                    # close enough → idle


class FlagSeekingPolicy:
    """Outputs [linear_x, angular_z] from ``obs['goal_vector']``.

    ``goal_vector`` is shape ``[B, 4]`` = ``[dx, dy, distance, bearing]`` in
    the rover's body frame, where ``bearing ∈ [-pi, pi]`` is the angle from
    the rover's heading to the goal (left positive).
    """

    def __init__(self, cfg: FlagSeekingPolicyConfig = None):
        self.cfg = cfg or FlagSeekingPolicyConfig()
        # Hysteresis: remember the offset we picked last frame and prefer
        # it over a slightly-better alternative. Without this, BEV noise
        # makes +25° and -25° each pass threshold on alternating frames
        # and the rover oscillates in place.
        self._last_offset_rad: float = 0.0
        # When the last offset still passes a *relaxed* threshold, keep it.
        self._hysteresis_obstacle_thresh: float = 0.70
        self._hysteresis_visible_frac: float = 0.05

    def compute(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        goal_vec = obs["goal_vector"]
        distance = goal_vec[..., 2]
        bearing = goal_vec[..., 3]

        omega = (self.cfg.kp_omega * bearing).clamp(
            -self.cfg.omega_max, self.cfg.omega_max)

        in_cone = bearing.abs() < self.cfg.align_cone_rad
        v = (self.cfg.v_max * torch.cos(bearing)).clamp_min(0.0)
        v = torch.where(in_cone, v, torch.zeros_like(v))

        close = distance < self.cfg.stop_distance_m
        v = torch.where(close, torch.zeros_like(v), v)
        omega = torch.where(close, torch.zeros_like(omega), omega)

        return torch.stack([v, omega], dim=-1)

    # Demo-source modes supported by ``compute_subgoal``.
    MODE_SIMPLE = "simple"            # straight at destination
    MODE_REACTIVE = "reactive"        # + BEV-aware sweep avoidance
    MODE_MEMORY = "memory"            # + visited-ground memory channel

    def compute_subgoal(
        self,
        obs: Dict[str, torch.Tensor],
        r_min: float = 2.0,
        r_max: float = 10.0,
        mode: str = "simple",
        bev_extent_m: float = 15.0,
    ) -> torch.Tensor:
        """Generate a BC-label sub-goal in rover-body polar coords.

        The base policy ALWAYS points straight at the current target.
        Slope / pit / stuck avoidance is handled by the driver as brief
        interrupt overrides, not by changing the base bearing. ``mode``
        is accepted for backwards-compatibility but ignored.
        """
        goal_vec = obs["goal_vector"]
        distance = goal_vec[..., 2]
        bearing = goal_vec[..., 3]
        r = distance.clamp(min=r_min, max=r_max)
        return torch.stack([bearing, r], dim=-1)

