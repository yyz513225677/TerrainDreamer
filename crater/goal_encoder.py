"""MLP encoder for goal + mission phase + previous action.

Inputs concatenate to [B, 8]:
    goal_vector [B, 4]   (dx, dy, distance, bearing)
    mission_phase [B, 2] one-hot {OUTBOUND, RETURN}
    prev_action [B, 2]   previous (linear_x, angular_z)

Output : [B, goal_embed]   (default 128)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalEncoder(nn.Module):
    def __init__(
        self,
        goal_dim: int = 4,
        phase_dim: int = 2,
        action_dim: int = 2,
        embed_dim: int = 128,
        hidden: int = 256,
    ):
        super().__init__()
        in_dim = goal_dim + phase_dim + action_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, embed_dim)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(
        self,
        goal: torch.Tensor,
        phase: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        if goal.dim() != 2 or phase.dim() != 2 or prev_action.dim() != 2:
            raise ValueError(
                f"GoalEncoder expects [B, D]; got "
                f"goal={tuple(goal.shape)}, phase={tuple(phase.shape)}, "
                f"prev_action={tuple(prev_action.shape)}")
        x = torch.cat([goal, phase, prev_action], dim=-1)
        x = F.silu(self.norm1(self.fc1(x)))
        x = F.silu(self.norm2(self.fc2(x)))
        return self.fc3(x)
