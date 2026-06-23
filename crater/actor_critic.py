"""Actor and Critic.

The actor outputs a tanh-squashed Gaussian over (linear_x, angular_z) which
is then scaled per-dim into the rover's valid command range:
    linear_x  in [action_low[0], action_high[0]]  (e.g. [0.0, 0.8])
    angular_z in [action_low[1], action_high[1]]  (e.g. [-1.0, 1.0])
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        action_dim: int = 2,
        actor_hidden: int = 512,
        critic_hidden: int = 512,
        action_low: Tuple[float, float] = (0.0, -1.0),
        action_high: Tuple[float, float] = (0.8, 1.0),
        min_std: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.min_std = min_std
        self.register_buffer("action_low", torch.tensor(action_low))
        self.register_buffer("action_high", torch.tensor(action_high))

        self.actor_trunk = nn.Sequential(
            nn.Linear(feat_dim, actor_hidden), nn.SiLU(),
            nn.Linear(actor_hidden, actor_hidden), nn.SiLU(),
            nn.Linear(actor_hidden, actor_hidden), nn.SiLU(),
        )
        self.actor_head = nn.Linear(actor_hidden, 2 * action_dim)

        self.critic_net = nn.Sequential(
            nn.Linear(feat_dim, critic_hidden), nn.SiLU(),
            nn.Linear(critic_hidden, critic_hidden), nn.SiLU(),
            nn.Linear(critic_hidden, critic_hidden), nn.SiLU(),
            nn.Linear(critic_hidden, 1),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Action scaling helpers
    # ──────────────────────────────────────────────────────────────────────
    def scale_action(self, tanh_action: torch.Tensor) -> torch.Tensor:
        """Map [-1, 1] (tanh output) -> per-dim [low, high]."""
        half_range = 0.5 * (self.action_high - self.action_low)
        center = 0.5 * (self.action_high + self.action_low)
        return center + half_range * tanh_action

    def unscale_action(self, scaled: torch.Tensor) -> torch.Tensor:
        """Map per-dim [low, high] -> [-1, 1] (inverse of scale_action)."""
        half_range = 0.5 * (self.action_high - self.action_low)
        center = 0.5 * (self.action_high + self.action_low)
        return ((scaled - center) / half_range).clamp(-0.999_999, 0.999_999)

    # ──────────────────────────────────────────────────────────────────────
    # Actor
    # ──────────────────────────────────────────────────────────────────────
    def actor(self, feat: torch.Tensor) -> Normal:
        """Return the pre-tanh-squash Gaussian over actions."""
        x = self.actor_trunk(feat)
        mean, std = self.actor_head(x).chunk(2, dim=-1)
        std = F.softplus(std) + self.min_std
        return Normal(mean, std)

    def critic(self, feat: torch.Tensor) -> torch.Tensor:
        return self.critic_net(feat).squeeze(-1)

    # ──────────────────────────────────────────────────────────────────────
    # Sampling
    # ──────────────────────────────────────────────────────────────────────
    def sample_action(
        self,
        feat: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Normal]:
        """Return (scaled_action, log_prob, pre_tanh_dist)."""
        dist = self.actor(feat)
        raw = dist.mean if deterministic else dist.rsample()
        tanh_a = torch.tanh(raw)
        scaled = self.scale_action(tanh_a)
        # log-prob with tanh + per-dim scaling Jacobian:
        half_range = 0.5 * (self.action_high - self.action_low)
        log_p = dist.log_prob(raw)
        log_det = torch.log(half_range * (1.0 - tanh_a.pow(2)) + 1e-6)
        log_prob = (log_p - log_det).sum(-1)
        return scaled, log_prob, dist

    def log_prob_of(self, dist: Normal, scaled_action: torch.Tensor) -> torch.Tensor:
        """Log-prob of an already-scaled action, accounting for tanh+scale Jacobian."""
        tanh_a = self.unscale_action(scaled_action)
        raw = torch.atanh(tanh_a)
        half_range = 0.5 * (self.action_high - self.action_low)
        log_p = dist.log_prob(raw)
        log_det = torch.log(half_range * (1.0 - tanh_a.pow(2)) + 1e-6)
        return (log_p - log_det).sum(-1)
