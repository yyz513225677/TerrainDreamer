"""Sequence replay buffer for Dreamer-style training.

Stores per-episode transitions and samples fixed-length windows for the
world-model update. Observations are dicts of tensors so the buffer
doesn't need to know the schema:

    obs = {
        'bev'  : tensor [4, 128, 128],
        'imu'  : tensor [12],
        'goal' : tensor [4],
        'phase': tensor [2],
        'prev_action': tensor [2],
    }
"""
from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Tuple

import torch


Obs = Dict[str, torch.Tensor]


class Transition:
    __slots__ = ("obs", "action", "reward", "done", "next_obs", "info")

    def __init__(self, obs: Obs, action: torch.Tensor, reward: float,
                 done: bool, next_obs: Obs, info: dict | None = None):
        self.obs = {k: v.detach().cpu() for k, v in obs.items()}
        self.action = action.detach().cpu()
        self.reward = float(reward)
        self.done = bool(done)
        self.next_obs = {k: v.detach().cpu() for k, v in next_obs.items()}
        self.info = info or {}


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        # Episodes are stored as lists of transitions; we sample windows
        # out of them so terminal boundaries are preserved.
        self._episodes: deque[List[Transition]] = deque()
        self._current: List[Transition] = []
        self._n_steps: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # Insertion
    # ──────────────────────────────────────────────────────────────────────
    def add(self, transition: Transition) -> None:
        self._current.append(transition)
        self._n_steps += 1
        if transition.done:
            self._episodes.append(self._current)
            self._current = []
        self._maybe_evict()

    def end_episode(self) -> None:
        """Force-close the current episode without an explicit `done`."""
        if self._current:
            self._episodes.append(self._current)
            self._current = []

    def _maybe_evict(self) -> None:
        while self._n_steps > self.capacity and self._episodes:
            dropped = self._episodes.popleft()
            self._n_steps -= len(dropped)

    # ──────────────────────────────────────────────────────────────────────
    # Sampling
    # ──────────────────────────────────────────────────────────────────────
    def can_sample(self, batch_size: int, seq_len: int) -> bool:
        # We need at least one episode with length >= seq_len.
        return any(len(ep) >= seq_len for ep in self._episodes) and \
               self._n_steps >= batch_size * seq_len

    def sample(self, batch_size: int, seq_len: int) -> Tuple[
        Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor,
        Dict[str, torch.Tensor]
    ]:
        """Sample a batch of contiguous windows.

        Returns:
            obs       : dict of tensors [B, T, ...]
            actions   : [B, T, A]
            rewards   : [B, T]
            dones     : [B, T]
            next_obs  : dict of tensors [B, T, ...]
        """
        eligible = [ep for ep in self._episodes if len(ep) >= seq_len]
        if not eligible:
            raise RuntimeError(
                "ReplayBuffer.sample: no episodes of length >= seq_len. "
                "Call can_sample() first.")

        windows: List[List[Transition]] = []
        for _ in range(batch_size):
            ep = random.choice(eligible)
            start = random.randint(0, len(ep) - seq_len)
            windows.append(ep[start:start + seq_len])
        return self._collate(windows)

    @staticmethod
    def _stack_obs(obs_list: List[Obs]) -> Dict[str, torch.Tensor]:
        keys = obs_list[0].keys()
        return {k: torch.stack([o[k] for o in obs_list], dim=0) for k in keys}

    def _collate(self, windows: List[List[Transition]]):
        B, T = len(windows), len(windows[0])
        obs_stacks: Dict[str, List[torch.Tensor]] = {}
        next_obs_stacks: Dict[str, List[torch.Tensor]] = {}
        actions = torch.zeros(B, T, windows[0][0].action.shape[0])
        rewards = torch.zeros(B, T)
        dones = torch.zeros(B, T)

        for b, window in enumerate(windows):
            for t, tr in enumerate(window):
                for k, v in tr.obs.items():
                    obs_stacks.setdefault(k, [None] * (B * T))
                    obs_stacks[k][b * T + t] = v
                for k, v in tr.next_obs.items():
                    next_obs_stacks.setdefault(k, [None] * (B * T))
                    next_obs_stacks[k][b * T + t] = v
                actions[b, t] = tr.action
                rewards[b, t] = tr.reward
                dones[b, t] = float(tr.done)

        obs = {k: torch.stack(v, dim=0).view(B, T, *v[0].shape)
               for k, v in obs_stacks.items()}
        next_obs = {k: torch.stack(v, dim=0).view(B, T, *v[0].shape)
                    for k, v in next_obs_stacks.items()}
        return obs, actions, rewards, dones, next_obs

    # ──────────────────────────────────────────────────────────────────────
    # Bookkeeping
    # ──────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return self._n_steps

    def num_episodes(self) -> int:
        return len(self._episodes)
