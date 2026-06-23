"""Outbound waypoint recorder + reversed return queue.

Waypoint format: (x, y, yaw). Stored every `waypoint_spacing_m` while in
the OUTBOUND phase, then reversed (excluding the destination — already
there) to form the return queue.

Pure Python; no torch dependency.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple


Waypoint = Tuple[float, float, float]   # (x, y, yaw)


class TrajectoryMemory:
    def __init__(self, waypoint_spacing_m: float = 0.5,
                 waypoint_reach_m: float = 0.5):
        self.waypoint_spacing_m = float(waypoint_spacing_m)
        self.waypoint_reach_m = float(waypoint_reach_m)
        self._outbound: List[Waypoint] = []
        self._return: List[Waypoint] = []
        self._finalized = False

    # ──────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        self._outbound.clear()
        self._return.clear()
        self._finalized = False

    # ──────────────────────────────────────────────────────────────────────
    # Outbound recording
    # ──────────────────────────────────────────────────────────────────────
    def add_waypoint(self, pose: Waypoint) -> bool:
        """Append (x, y, yaw) if rover has moved >= waypoint_spacing_m.

        Returns True iff the waypoint was actually stored.
        Ignored once `finalize_outbound` has been called.
        """
        if self._finalized:
            return False
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        if not self._outbound:
            self._outbound.append((x, y, yaw))
            return True
        last_x, last_y, _ = self._outbound[-1]
        if math.hypot(x - last_x, y - last_y) >= self.waypoint_spacing_m:
            self._outbound.append((x, y, yaw))
            return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Phase switch
    # ──────────────────────────────────────────────────────────────────────
    def finalize_outbound(self) -> None:
        """Freeze outbound trail and build reversed return queue.

        Excludes the final outbound pose (rover is already there) and
        terminates with the origin so the rover homes in on the start pose.
        """
        if self._finalized:
            return
        if len(self._outbound) >= 2:
            self._return = list(reversed(self._outbound[:-1]))
        else:
            self._return = list(self._outbound)
        self._finalized = True

    # ──────────────────────────────────────────────────────────────────────
    # Return-phase queries
    # ──────────────────────────────────────────────────────────────────────
    def get_current_return_waypoint(self) -> Optional[Waypoint]:
        return self._return[0] if self._return else None

    def advance_if_reached(self, current_pose: Waypoint) -> bool:
        """Pop the current return waypoint if the rover is within `waypoint_reach_m`."""
        if not self._return:
            return False
        wx, wy, _ = self._return[0]
        if math.hypot(current_pose[0] - wx, current_pose[1] - wy) <= self.waypoint_reach_m:
            self._return.pop(0)
            return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Read-only views
    # ──────────────────────────────────────────────────────────────────────
    def get_outbound_path(self) -> List[Waypoint]:
        return list(self._outbound)

    def get_reversed_path(self) -> List[Waypoint]:
        return list(self._return)

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    def __len__(self) -> int:
        return len(self._outbound)
