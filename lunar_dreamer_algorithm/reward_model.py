"""Reward predictor — scalar reward from RSSM feature."""
from __future__ import annotations

import torch
import torch.nn as nn


class RewardModel(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),   nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)
