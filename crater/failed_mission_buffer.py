"""FailedMissionBuffer — records missions that ended in failure.

Stored fields per failed mission:
    mission_id                int
    initial_pose              (x, y, yaw)
    destination               (x, y)
    mission_phase_at_failure  str           'OUTBOUND' | 'RETURN'
    outbound_trajectory       list of (x, y, yaw)
    recent_observation_sequence  list of dict[str, Tensor]  (B-less)
    recent_action_sequence       list of Tensor             (shape [A])
    recent_reward_sequence       list of float
    failure_reason            str           'collision' | 'timeout' | ...
    failure_pose              (x, y, yaw)
    current_target            (x, y)
    episode_index             int
    recovered                 bool          (set later when recovery succeeds)

The buffer is a bounded FIFO; oldest non-recovered missions are evicted
when at capacity.
"""
from __future__ import annotations

import random
import threading
from collections import deque
from typing import Any, Dict, List, Optional


# Allowed failure reasons (free-form strings are also tolerated).
FAILURE_REASONS = (
    "collision",
    "timeout",
    "stuck",
    "excessive_roll",
    "excessive_pitch",
    "failed_to_reach_destination",
    "failed_to_return_to_origin",
    "out_of_map",
)


class FailedMissionBuffer:
    def __init__(self, max_failed_missions: int = 256):
        self.capacity = int(max_failed_missions)
        self._records: deque = deque()           # ordered insertion
        self._by_id: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    # ──────────────────────────────────────────────────────────────────────
    def add_failed_mission(self, mission_data: Dict[str, Any]) -> int:
        """Insert a failure record, returns its assigned mission_id.

        Required keys (other keys are also retained):
            initial_pose, destination, mission_phase_at_failure,
            outbound_trajectory, recent_observation_sequence,
            recent_action_sequence, recent_reward_sequence,
            failure_reason, failure_pose, current_target, episode_index
        """
        required = (
            "initial_pose", "destination", "mission_phase_at_failure",
            "outbound_trajectory", "recent_observation_sequence",
            "recent_action_sequence", "recent_reward_sequence",
            "failure_reason", "failure_pose", "current_target",
            "episode_index",
        )
        missing = [k for k in required if k not in mission_data]
        if missing:
            raise KeyError(
                f"FailedMissionBuffer.add_failed_mission missing keys: {missing}")

        with self._lock:
            mid = self._next_id
            self._next_id += 1
            record = {"mission_id": mid, "recovered": False, **mission_data}
            self._records.append(record)
            self._by_id[mid] = record
            self._evict_locked()
            return mid

    def _evict_locked(self) -> None:
        # Drop oldest non-recovered when over capacity.
        while len(self._records) > self.capacity:
            old = self._records.popleft()
            self._by_id.pop(old["mission_id"], None)

    # ──────────────────────────────────────────────────────────────────────
    def sample_failed_mission(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._records:
                return None
            return random.choice(self._records)

    def get_latest_failed_mission(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._records[-1] if self._records else None

    # ──────────────────────────────────────────────────────────────────────
    def mark_recovered(self, mission_id: int) -> bool:
        with self._lock:
            rec = self._by_id.get(int(mission_id))
            if rec is None:
                return False
            rec["recovered"] = True
            return True

    def clear_recovered(self) -> int:
        with self._lock:
            keep: deque = deque()
            removed = 0
            for r in self._records:
                if r["recovered"]:
                    self._by_id.pop(r["mission_id"], None)
                    removed += 1
                else:
                    keep.append(r)
            self._records = keep
            return removed

    def list_active(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [r for r in self._records if not r["recovered"]]

    # ──────────────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        import torch  # local import keeps load-time torch optional for tests
        with self._lock:
            torch.save({"records": list(self._records),
                        "next_id": self._next_id}, path)

    def load(self, path: str) -> None:
        import torch
        state = torch.load(path, map_location="cpu")
        with self._lock:
            self._records = deque(state["records"])
            self._by_id = {r["mission_id"]: r for r in self._records}
            self._next_id = int(state["next_id"])
