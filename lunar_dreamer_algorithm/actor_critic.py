"""Actor and critic networks.

Actor outputs a tanh-squashed Gaussian over (linear_x, angular_z).
The final action is scaled into the per-dim ranges:
    linear_x  in [action_low[0], action_high[0]]  (e.g. [0.0, 0.8])
    angular_z in [action_low[1], action_high[1]]  (e.g. [-1.0, 1.0])
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        action_dim: int = 2,
        hidden: int = 512,
        action_low: Tuple[float, float] = (0.0, -1.0),
        action_high: Tuple[float, float] = (0.8, 1.0),
        min_std: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.min_std = min_std
        self.register_buffer("action_low", torch.tensor(action_low))
        self.register_buffer("action_high", torch.tensor(action_high))

        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),   nn.SiLU(),
            nn.Linear(hidden, hidden),   nn.SiLU(),
        )
        self.head = nn.Linear(hidden, 2 * action_dim)

    def _scale(self, tanh_action: torch.Tensor) -> torch.Tensor:
        """Map [-1, 1] -> [low, high] per dimension."""
        half_range = 0.5 * (self.action_high - self.action_low)
        center = 0.5 * (self.action_high + self.action_low)
        return center + half_range * tanh_action

    def forward(self, feat: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """Return (Normal in pre-tanh space, scaled action sample)."""
        x = self.net(feat)
        mean, std = self.head(x).chunk(2, dim=-1)
        std = F.softplus(std) + self.min_std
        dist = Normal(mean, std)
        raw = dist.rsample()
        scaled = self._scale(torch.tanh(raw))
        return dist, scaled

    def act(self, feat: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Convenience inference call."""
        x = self.net(feat)
        mean, std = self.head(x).chunk(2, dim=-1)
        std = F.softplus(std) + self.min_std
        raw = mean if deterministic else Normal(mean, std).sample()
        return self._scale(torch.tanh(raw))

    def log_prob(self, dist: Normal, scaled_action: torch.Tensor) -> torch.Tensor:
        """Log-prob accounting for tanh squashing and per-dim scaling.

        Inverts: scaled = center + half_range * tanh(raw) -> tanh(raw) = (s-c)/hr
        Log-det-Jacobian: log(half_range * (1 - tanh^2))
        """
        half_range = 0.5 * (self.action_high - self.action_low)
        center = 0.5 * (self.action_high + self.action_low)
        tanh_a = ((scaled_action - center) / half_range).clamp(-0.999_999, 0.999_999)
        raw = torch.atanh(tanh_a)
        log_p = dist.log_prob(raw)
        log_det = torch.log(half_range * (1.0 - tanh_a.pow(2)) + 1e-6)
        return (log_p - log_det).sum(-1)


class Critic(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),   nn.SiLU(),
            nn.Linear(hidden, hidden),   nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)
