"""Fuse terrain + IMU + goal embeddings into a single observation embedding.

Output : [B, obs_embed]   (default 768)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionEncoder(nn.Module):
    def __init__(
        self,
        terrain_embed: int = 512,
        imu_embed: int = 128,
        goal_embed: int = 128,
        obs_embed: int = 768,
    ):
        super().__init__()
        in_dim = terrain_embed + imu_embed + goal_embed
        self.proj = nn.Linear(in_dim, obs_embed)
        self.norm = nn.LayerNorm(obs_embed)

    def forward(
        self,
        terrain: torch.Tensor,
        imu: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([terrain, imu, goal], dim=-1)
        return F.silu(self.norm(self.proj(x)))
