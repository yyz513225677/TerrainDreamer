"""
Sequence Replay Buffer for Dreamer World Model Training
=========================================================
Stores complete episodes as sequences of:
  - point cloud features  [T, N, 8]   (precomputed by PointCloudProcessor)
  - actions               [T, 2]
  - rewards               [T]
  - continues             [T]  (1 = not terminal, 0 = terminal)

Sampling returns fixed-length sub-sequences of length `seq_len` drawn
uniformly across all stored steps, matching the DreamerV3 replay strategy.

Thread-safety: not thread-safe (single-process training).

Memory layout
-------------
  Internal storage is a list of episode dicts (variable length).
  When `max_episodes` is reached, oldest episodes are evicted.
  `sample()` draws from a flat index over all stored (episode, start) pairs
  so that every step has equal probability.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, Optional, List


class DreamerReplayBuffer:
    """
    Replay buffer for online Dreamer world model training.

    Parameters
    ----------
    seq_len : int
        Length of each sampled sub-sequence (T in the batch).
    max_points : int
        Max point cloud points per step (N dimension).
    feat_dim : int
        Point cloud feature dimension (C = 8 for standard pipeline).
    action_dim : int
        Action dimension (2 for Jackal lin/ang).
    max_episodes : int
        Capacity in number of episodes (oldest are evicted when full).
    min_episodes : int
        Minimum episodes stored before sampling is allowed.
    """

    def __init__(
        self,
        seq_len: int = 16,
        max_points: int = 1024,
        feat_dim: int = 8,
        action_dim: int = 2,
        max_episodes: int = 500,
        min_episodes: int = 5,
    ):
        self.seq_len      = seq_len
        self.max_points   = max_points
        self.feat_dim     = feat_dim
        self.action_dim   = action_dim
        self.max_episodes = max_episodes
        self.min_episodes = min_episodes

        self._episodes: deque = deque()
        self._total_steps: int = 0

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def add_episode(
        self,
        features:  np.ndarray,                   # [T, N, 8]  float32
        actions:   np.ndarray,                   # [T, 2]     float32
        rewards:   np.ndarray,                   # [T]        float32
        continues: np.ndarray,                   # [T]        float32  (1 / 0)
        goal_obs:  Optional[np.ndarray] = None,  # [T, 4]    float32  (goal in robot frame)
    ) -> bool:
        """Store one complete episode. Evicts oldest if at capacity.
        Returns False (and skips) if episode is shorter than seq_len."""
        T = len(actions)
        if T < self.seq_len:
            return False   # too short — no valid sub-sequences possible
        assert features.shape  == (T, self.max_points, self.feat_dim), \
            f"features shape mismatch: {features.shape}"
        assert actions.shape   == (T, self.action_dim)
        assert rewards.shape   == (T,)
        assert continues.shape == (T,)

        ep = {
            "features":  features.astype(np.float32),
            "actions":   actions.astype(np.float32),
            "rewards":   rewards.astype(np.float32),
            "continues": continues.astype(np.float32),
            "goal_obs":  (goal_obs.astype(np.float32)
                          if goal_obs is not None
                          else np.zeros((T, 4), dtype=np.float32)),
            "length":    T,
        }
        if len(self._episodes) >= self.max_episodes:
            evicted = self._episodes.popleft()
            self._total_steps -= evicted["length"]

        self._episodes.append(ep)
        self._total_steps += T
        return True

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def ready(self) -> bool:
        """True when enough data has been collected to start training."""
        if len(self._episodes) < self.min_episodes:
            return False
        # Need at least one episode with length >= seq_len
        return any(ep["length"] >= self.seq_len for ep in self._episodes)

    def __len__(self) -> int:
        return self._total_steps

    def num_episodes(self) -> int:
        return len(self._episodes)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a batch of fixed-length sub-sequences.

        Returns dict with keys:
          features  [B, T, N, 8]
          actions   [B, T, 2]
          rewards   [B, T]
          continues [B, T]
        """
        assert self.ready(), "Buffer not ready — collect more episodes first"

        # Build flat index of valid (episode_idx, start_t) pairs
        # A start_t is valid if episode has >= seq_len steps remaining
        index = []
        for ep_idx, ep in enumerate(self._episodes):
            max_start = ep["length"] - self.seq_len
            if max_start >= 0:
                for t in range(max_start + 1):
                    index.append((ep_idx, t))

        if len(index) == 0:
            raise RuntimeError(
                f"No valid sub-sequences: all {len(self._episodes)} episodes "
                f"are shorter than seq_len={self.seq_len}"
            )

        chosen = [index[i] for i in
                  np.random.choice(len(index), size=batch_size, replace=True)]

        feat_b  = np.empty((batch_size, self.seq_len, self.max_points, self.feat_dim),
                           dtype=np.float32)
        act_b   = np.empty((batch_size, self.seq_len, self.action_dim), dtype=np.float32)
        rew_b   = np.empty((batch_size, self.seq_len), dtype=np.float32)
        cont_b  = np.empty((batch_size, self.seq_len), dtype=np.float32)
        goal_b  = np.empty((batch_size, self.seq_len, 4), dtype=np.float32)

        eps = list(self._episodes)
        for b, (ep_idx, t0) in enumerate(chosen):
            ep = eps[ep_idx]
            sl = slice(t0, t0 + self.seq_len)
            feat_b[b]  = ep["features"][sl]
            act_b[b]   = ep["actions"][sl]
            rew_b[b]   = ep["rewards"][sl]
            cont_b[b]  = ep["continues"][sl]
            goal_b[b]  = ep["goal_obs"][sl]

        return {
            "features":  feat_b,
            "actions":   act_b,
            "rewards":   rew_b,
            "continues": cont_b,
            "goal_obs":  goal_b,
        }

    # ------------------------------------------------------------------
    # Persistence (save/load between sessions)
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save buffer to a .npz file."""
        eps = list(self._episodes)
        np.savez_compressed(
            path,
            n_episodes=len(eps),
            **{f"feat_{i}":  eps[i]["features"]  for i in range(len(eps))},
            **{f"act_{i}":   eps[i]["actions"]   for i in range(len(eps))},
            **{f"rew_{i}":   eps[i]["rewards"]   for i in range(len(eps))},
            **{f"cont_{i}":  eps[i]["continues"] for i in range(len(eps))},
            **{f"goal_{i}":  eps[i]["goal_obs"]  for i in range(len(eps))},
        )

    def load(self, path: str):
        """Load buffer from a .npz file (appends to existing episodes)."""
        data = np.load(path)
        n = int(data["n_episodes"])
        for i in range(n):
            self.add_episode(
                features  = data[f"feat_{i}"],
                actions   = data[f"act_{i}"],
                rewards   = data[f"rew_{i}"],
                continues = data[f"cont_{i}"],
                goal_obs  = data[f"goal_{i}"] if f"goal_{i}" in data else None,
            )
