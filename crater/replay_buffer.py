"""Episode-aware sequence replay buffer.

Transitions store dict observations + control-mode metadata. The trainer
samples fixed-length windows for Dreamer + BC + recovery-BC losses.

Per-transition info schema:
    info['control_mode']             'autonomous' | 'human'
    info['demo_type']                'normal' | 'failure_recovery'
    info['source_failed_mission_id'] Optional[int]
"""
from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch


Obs = Dict[str, torch.Tensor]

CONTROL_MODE_AUTO = "autonomous"
CONTROL_MODE_HUMAN = "human"
DEMO_TYPE_NORMAL = "normal"
DEMO_TYPE_RECOVERY = "failure_recovery"


class Transition:
    __slots__ = ("obs", "action", "reward", "done", "next_obs", "info")

    def __init__(
        self,
        obs: Obs,
        action: torch.Tensor,
        reward: float,
        done: bool,
        next_obs: Obs,
        info: Optional[Dict[str, Any]] = None,
    ):
        self.obs = {k: v.detach().cpu() for k, v in obs.items()}
        self.action = action.detach().cpu()
        self.reward = float(reward)
        self.done = bool(done)
        self.next_obs = {k: v.detach().cpu() for k, v in next_obs.items()}
        info = dict(info) if info else {}
        info.setdefault("control_mode", CONTROL_MODE_AUTO)
        info.setdefault("demo_type", DEMO_TYPE_NORMAL)
        info.setdefault("source_failed_mission_id", None)
        self.info = info


# ────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        # Stored as list-of-episodes so window sampling respects episode
        # boundaries.
        self._episodes: deque[List[Transition]] = deque()
        self._current: List[Transition] = []
        self._n_steps: int = 0

    # ──────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return self._n_steps

    def num_episodes(self) -> int:
        return len(self._episodes)

    # ──────────────────────────────────────────────────────────────────────
    def add(self, transition: Transition) -> None:
        self._current.append(transition)
        self._n_steps += 1
        if transition.done:
            self._episodes.append(self._current)
            self._current = []
        self._maybe_evict()

    def end_episode(self) -> None:
        """Force-close the in-progress episode without an explicit `done`."""
        if self._current:
            self._episodes.append(self._current)
            self._current = []

    def _maybe_evict(self) -> None:
        while self._n_steps > self.capacity and self._episodes:
            dropped = self._episodes.popleft()
            self._n_steps -= len(dropped)

    # ──────────────────────────────────────────────────────────────────────
    # Eligibility predicates
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _has_mode(window: List[Transition], control_mode: str) -> bool:
        return any(t.info.get("control_mode") == control_mode for t in window)

    @staticmethod
    def _has_demo_type(window: List[Transition], demo_type: str) -> bool:
        return any(t.info.get("demo_type") == demo_type for t in window)

    # ──────────────────────────────────────────────────────────────────────
    # Sampling
    # ──────────────────────────────────────────────────────────────────────
    def _candidate_windows(
        self,
        seq_len: int,
        require_mode: Optional[str] = None,
        require_demo_type: Optional[str] = None,
    ) -> List[Tuple[List[Transition], int]]:
        """Return list of (episode, start_index) pairs eligible for sampling."""
        candidates = []
        for ep in self._episodes:
            if len(ep) < seq_len:
                continue
            if require_mode is None and require_demo_type is None:
                # Any window inside the episode is OK.
                candidates.append((ep, None))
                continue
            # Filter on a per-window basis.
            for start in range(0, len(ep) - seq_len + 1):
                window = ep[start:start + seq_len]
                if require_mode and not self._has_mode(window, require_mode):
                    continue
                if require_demo_type and not self._has_demo_type(window, require_demo_type):
                    continue
                candidates.append((ep, start))
        return candidates

    def _sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        require_mode: Optional[str] = None,
        require_demo_type: Optional[str] = None,
    ):
        cands = self._candidate_windows(seq_len, require_mode, require_demo_type)
        if not cands:
            return None
        windows = []
        for _ in range(batch_size):
            ep, fixed_start = random.choice(cands)
            if fixed_start is None:
                start = random.randint(0, len(ep) - seq_len)
            else:
                start = fixed_start
            windows.append(ep[start:start + seq_len])
        return self._collate(windows)

    def can_sample(self, batch_size: int, seq_len: int) -> bool:
        return any(len(ep) >= seq_len for ep in self._episodes) and \
               self._n_steps >= batch_size * seq_len

    def sample_sequence_batch(self, batch_size: int, seq_len: int):
        out = self._sample_batch(batch_size, seq_len)
        if out is None:
            raise RuntimeError("ReplayBuffer.sample_sequence_batch: no eligible windows.")
        return out

    def sample_human_demo_batch(self, batch_size: int, seq_len: int):
        return self._sample_batch(batch_size, seq_len,
                                  require_mode=CONTROL_MODE_HUMAN)

    def sample_recovery_demo_batch(self, batch_size: int, seq_len: int):
        return self._sample_batch(batch_size, seq_len,
                                  require_demo_type=DEMO_TYPE_RECOVERY)

    def sample_mixed_batch(
        self,
        batch_size: int,
        seq_len: int,
        recovery_priority: float = 0.5,
    ):
        """Mix recovery demos and normal transitions in one batch."""
        n_recov = max(1, int(round(batch_size * recovery_priority)))
        n_normal = max(0, batch_size - n_recov)
        recov = self.sample_recovery_demo_batch(n_recov, seq_len)
        normal = self._sample_batch(n_normal, seq_len) if n_normal > 0 else None
        if recov is None:
            return normal
        if normal is None:
            return recov
        return self._concat(recov, normal)

    # ──────────────────────────────────────────────────────────────────────
    # Collation
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _concat(a, b):
        obs_a, act_a, rew_a, done_a, next_a = a
        obs_b, act_b, rew_b, done_b, next_b = b
        cat = lambda x, y: torch.cat([x, y], dim=0)
        obs = {k: cat(obs_a[k], obs_b[k]) for k in obs_a}
        next_obs = {k: cat(next_a[k], next_b[k]) for k in next_a}
        return obs, cat(act_a, act_b), cat(rew_a, rew_b), cat(done_a, done_b), next_obs

    def _collate(self, windows: List[List[Transition]]):
        B, T = len(windows), len(windows[0])
        obs_buf: Dict[str, List[torch.Tensor]] = {}
        next_obs_buf: Dict[str, List[torch.Tensor]] = {}
        actions = torch.zeros(B, T, windows[0][0].action.shape[0])
        rewards = torch.zeros(B, T)
        dones = torch.zeros(B, T)

        for b, window in enumerate(windows):
            for t, tr in enumerate(window):
                for k, v in tr.obs.items():
                    obs_buf.setdefault(k, [None] * (B * T))
                    obs_buf[k][b * T + t] = v
                for k, v in tr.next_obs.items():
                    next_obs_buf.setdefault(k, [None] * (B * T))
                    next_obs_buf[k][b * T + t] = v
                actions[b, t] = tr.action
                rewards[b, t] = tr.reward
                dones[b, t] = float(tr.done)

        obs = {k: torch.stack(v, dim=0).view(B, T, *v[0].shape) for k, v in obs_buf.items()}
        next_obs = {k: torch.stack(v, dim=0).view(B, T, *v[0].shape) for k, v in next_obs_buf.items()}
        return obs, actions, rewards, dones, next_obs
