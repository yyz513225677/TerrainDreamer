"""Continuation predictor — Bernoulli probability that the episode continues."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Bernoulli


class ContinuationModel(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),   nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Return continuation logit (use .dist for sampling/log-prob)."""
        return self.net(feat).squeeze(-1)

    def dist(self, feat: torch.Tensor) -> Bernoulli:
        return Bernoulli(logits=self.forward(feat))
