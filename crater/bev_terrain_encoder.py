"""CNN encoder for the LiDAR BEV terrain tensor.

Input  : [B, 4, 128, 128]
         channels = (occupancy, max_height, elevation, roughness)
Output : [B, terrain_embed]   (default 512)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVTerrainEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, embed_dim: int = 512):
        super().__init__()
        # 128 -> 64 -> 32 -> 16 -> 8 with stride-2 convolutions.
        self.conv1 = nn.Conv2d(in_channels, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 64)
        self.norm3 = nn.GroupNorm(8, 128)
        self.norm4 = nn.GroupNorm(8, 256)
        self.fc = nn.Linear(256 * 8 * 8, embed_dim)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        if bev.dim() != 4 or bev.shape[1] != 4:
            raise ValueError(
                f"BEVTerrainEncoder expects [B, 4, H, W]; got {tuple(bev.shape)}")
        x = F.silu(self.norm1(self.conv1(bev)))
        x = F.silu(self.norm2(self.conv2(x)))
        x = F.silu(self.norm3(self.conv3(x)))
        x = F.silu(self.norm4(self.conv4(x)))
        x = x.flatten(1)
        return self.fc(x)
