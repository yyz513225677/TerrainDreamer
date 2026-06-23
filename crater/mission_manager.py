"""Mission FSM: OUTBOUND → RETURN → SUCCESS.

The manager does *not* own a TrajectoryMemory; the caller passes one in via
`update()` so a single TrajectoryMemory can also be used elsewhere (e.g. for
the trainer's BC dataset). Phase transitions are driven by the rover's pose
which must be supplied in the world (or ground-truth) frame.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Tuple

from .trajectory_memory import TrajectoryMemory


Pose = Tuple[float, float, float]    # (x, y, yaw)


class MissionPhase(Enum):
    OUTBOUND = 0
    RETURN = 1
    SUCCESS = 2


class MissionManager:
    def __init__(
        self,
        destination_reach_m: float = 0.8,
        origin_reach_m: float = 0.8,
    ):
        self.destination_reach_m = float(destination_reach_m)
        self.origin_reach_m = float(origin_reach_m)
        self.phase: MissionPhase = MissionPhase.OUTBOUND
        self.origin: Tuple[float, float] = (0.0, 0.0)
        self.destination: Tuple[float, float] = (0.0, 0.0)
        self._current_target: Tuple[float, float] = (0.0, 0.0)
        self._success = False

    # ──────────────────────────────────────────────────────────────────────
    def reset(self, origin: Tuple[float, float],
              destination: Tuple[float, float]) -> None:
        self.origin = (float(origin[0]), float(origin[1]))
        self.destination = (float(destination[0]), float(destination[1]))
        self._current_target = self.destination
        self.phase = MissionPhase.OUTBOUND
        self._success = False

    # ──────────────────────────────────────────────────────────────────────
    def update(self, current_pose: Pose,
               trajectory_memory: TrajectoryMemory) -> None:
        """Advance the FSM with the latest pose. Records outbound waypoints
        and pops reached return waypoints via the supplied TrajectoryMemory."""
        x, y = current_pose[0], current_pose[1]

        if self.phase == MissionPhase.OUTBOUND:
            trajectory_memory.add_waypoint(current_pose)
            if math.hypot(x - self.destination[0], y - self.destination[1]) \
                    < self.destination_reach_m:
                trajectory_memory.finalize_outbound()
                self.phase = MissionPhase.RETURN

        if self.phase == MissionPhase.RETURN:
            # Symmetric with OUTBOUND: aim straight at the origin and let
            # the base policy's reactive sweep + stuck recovery handle
            # obstacles. Trajectory waypoints are no longer consulted —
            # the rover treats (0, 0) as just another goal.
            self._current_target = self.origin
            d_origin = math.hypot(x - self.origin[0], y - self.origin[1])
            if d_origin < self.origin_reach_m:
                self.phase = MissionPhase.SUCCESS
                self._success = True
                return

        if self.phase == MissionPhase.OUTBOUND:
            self._current_target = self.destination

    # ──────────────────────────────────────────────────────────────────────
    def get_current_target(self) -> Tuple[float, float]:
        return self._current_target

    def get_phase_vector(self) -> Tuple[float, float]:
        """One-hot [outbound, return]. SUCCESS reports as RETURN."""
        if self.phase == MissionPhase.OUTBOUND:
            return (1.0, 0.0)
        return (0.0, 1.0)

    def is_success(self) -> bool:
        return self._success

    def is_outbound(self) -> bool:
        return self.phase == MissionPhase.OUTBOUND

    def is_return(self) -> bool:
        return self.phase == MissionPhase.RETURN
