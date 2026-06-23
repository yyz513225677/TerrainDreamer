"""MLP encoder for the IMU vector.

Input  : [B, 12]   (lin_acc xyz, ang_vel xyz, roll, pitch, yaw_rate, vx, vy, vz)
Output : [B, imu_embed]   (default 128)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IMUEncoder(nn.Module):
    def __init__(self, imu_dim: int = 12, embed_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(imu_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, embed_dim)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        if imu.dim() != 2:
            raise ValueError(
                f"IMUEncoder expects [B, D]; got {tuple(imu.shape)}")
        x = F.silu(self.norm1(self.fc1(imu)))
        x = F.silu(self.norm2(self.fc2(x)))
        return self.fc3(x)
