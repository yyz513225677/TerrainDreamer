"""Records outbound waypoints and replays them in reverse on the way home.

Pure Python — no torch dependency. Pose inputs are (x, y) tuples or
length-2 numeric sequences in the world frame.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple


Point = Tuple[float, float]


class TrajectoryMemory:
    def __init__(self, waypoint_spacing_m: float = 0.5,
                 waypoint_reach_m: float = 0.5):
        self.waypoint_spacing_m = float(waypoint_spacing_m)
        self.waypoint_reach_m = float(waypoint_reach_m)
        self._outbound: List[Point] = []   # ordered outbound trail
        self._return: List[Point] = []     # reversed; index 0 = next target
        self._reversed = False

    # ──────────────────────────────────────────────────────────────────────
    # Outbound recording
    # ──────────────────────────────────────────────────────────────────────
    def record(self, x: float, y: float) -> bool:
        """Add a waypoint if we've moved >= waypoint_spacing_m from the last.

        Returns True if the point was actually stored.
        """
        if self._reversed:
            return False
        if not self._outbound:
            self._outbound.append((float(x), float(y)))
            return True
        last_x, last_y = self._outbound[-1]
        if math.hypot(x - last_x, y - last_y) >= self.waypoint_spacing_m:
            self._outbound.append((float(x), float(y)))
            return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Phase switch
    # ──────────────────────────────────────────────────────────────────────
    def reverse(self) -> None:
        """Freeze the outbound trail and prepare the reversed return queue.

        The queue excludes the destination itself (we're already there) and
        ends with the origin so the rover homes onto the starting pose.
        """
        if self._reversed:
            return
        if len(self._outbound) >= 2:
            self._return = list(reversed(self._outbound[:-1]))
        else:
            self._return = list(self._outbound)
        self._reversed = True

    def reset(self) -> None:
        self._outbound.clear()
        self._return.clear()
        self._reversed = False

    # ──────────────────────────────────────────────────────────────────────
    # Return-phase queries
    # ──────────────────────────────────────────────────────────────────────
    def current_waypoint(self) -> Optional[Point]:
        return self._return[0] if self._return else None

    def advance_if_close(self, x: float, y: float) -> bool:
        """Pop the current return waypoint once the rover is within
        `waypoint_reach_m` of it. Returns True if a waypoint was popped."""
        if not self._return:
            return False
        wx, wy = self._return[0]
        if math.hypot(x - wx, y - wy) <= self.waypoint_reach_m:
            self._return.pop(0)
            return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Introspection
    # ──────────────────────────────────────────────────────────────────────
    @property
    def outbound(self) -> List[Point]:
        return list(self._outbound)

    @property
    def remaining_return(self) -> List[Point]:
        return list(self._return)

    @property
    def is_reversed(self) -> bool:
        return self._reversed

    def __len__(self) -> int:
        return len(self._outbound)
