"""Mission FSM: OUTBOUND -> RETURN -> SUCCESS.

State transitions key off the rover's *current pose*. The caller is
responsible for feeding ground-truth or fused-localisation poses — this
module makes no assumptions about the source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .trajectory_memory import TrajectoryMemory


class MissionPhase(Enum):
    OUTBOUND = 0
    RETURN = 1
    SUCCESS = 2


@dataclass
class MissionStatus:
    phase: MissionPhase
    target: Tuple[float, float]
    distance_to_target: float
    bearing_to_target: float           # radians, world frame, from +X axis


class MissionManager:
    def __init__(
        self,
        destination: Tuple[float, float],
        origin: Tuple[float, float] = (0.0, 0.0),
        destination_reach_m: float = 0.8,
        origin_reach_m: float = 0.8,
        waypoint_spacing_m: float = 0.5,
        waypoint_reach_m: float = 0.5,
    ):
        self.destination = (float(destination[0]), float(destination[1]))
        self.origin = (float(origin[0]), float(origin[1]))
        self.destination_reach_m = float(destination_reach_m)
        self.origin_reach_m = float(origin_reach_m)
        self.phase = MissionPhase.OUTBOUND
        self.memory = TrajectoryMemory(
            waypoint_spacing_m=waypoint_spacing_m,
            waypoint_reach_m=waypoint_reach_m,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Phase update
    # ──────────────────────────────────────────────────────────────────────
    def update(self, x: float, y: float) -> MissionStatus:
        """Step the FSM with the latest robot pose, return current status."""
        if self.phase == MissionPhase.OUTBOUND:
            self.memory.record(x, y)
            d = math.hypot(x - self.destination[0], y - self.destination[1])
            if d < self.destination_reach_m:
                self.memory.reverse()
                self.phase = MissionPhase.RETURN

        if self.phase == MissionPhase.RETURN:
            # Pop reached waypoints, then check for global success.
            while self.memory.advance_if_close(x, y):
                pass
            d_origin = math.hypot(x - self.origin[0], y - self.origin[1])
            if d_origin < self.origin_reach_m:
                self.phase = MissionPhase.SUCCESS

        return self.status(x, y)

    # ──────────────────────────────────────────────────────────────────────
    # Target lookup
    # ──────────────────────────────────────────────────────────────────────
    def current_target(self) -> Tuple[float, float]:
        if self.phase == MissionPhase.OUTBOUND:
            return self.destination
        if self.phase == MissionPhase.RETURN:
            wp = self.memory.current_waypoint()
            return wp if wp is not None else self.origin
        return self.origin  # SUCCESS — stay at origin

    def status(self, x: float, y: float) -> MissionStatus:
        target = self.current_target()
        dx = target[0] - x
        dy = target[1] - y
        return MissionStatus(
            phase=self.phase,
            target=target,
            distance_to_target=math.hypot(dx, dy),
            bearing_to_target=math.atan2(dy, dx),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Convenience for observation packing
    # ──────────────────────────────────────────────────────────────────────
    def phase_one_hot(self) -> Tuple[float, float]:
        """[outbound, return] one-hot. SUCCESS is reported as RETURN since
        from the policy's perspective the mission is over."""
        if self.phase == MissionPhase.OUTBOUND:
            return (1.0, 0.0)
        return (0.0, 1.0)

    def is_done(self) -> bool:
        return self.phase == MissionPhase.SUCCESS

    def reset(self, destination: Optional[Tuple[float, float]] = None) -> None:
        if destination is not None:
            self.destination = (float(destination[0]), float(destination[1]))
        self.phase = MissionPhase.OUTBOUND
        self.memory.reset()
