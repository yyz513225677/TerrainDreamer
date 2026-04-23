"""
Terrain Encoder
================
Encodes processed point clouds into fixed-size latent feature vectors.
Uses a PointPillars architecture (BEV pseudo-image + 2D conv backbone).

Input:  [B, N, C] point features (x, y, z, intensity, nx, ny, nz, height)
Output: [B, feature_dim] latent terrain embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class TerrainEncoder(nn.Module):
    """
    PointPillars-based terrain encoder.

    Pipeline:
      1. Assign each point to a pillar (xy grid, infinite z)
      2. Augment per-point features with offsets to pillar mean and center
      3. Pillar Feature Net (linear → BN → ReLU)
      4. Scatter max to BEV pseudo-image [B, C, H, W]
      5. 2D conv backbone
      6. Global pool → MLP → feature_dim
    """

    def __init__(
        self,
        input_channels: int = 8,
        feature_dim: int = 256,
        pillar_size: float = 2.0,
        x_range: Tuple[float, float] = (-60.0, 60.0),
        y_range: Tuple[float, float] = (-60.0, 60.0),
        pillar_feat_dim: int = 64,
        # Accepted for backward compat with PointNet++ kwargs (ignored)
        sa_npoints: List[int] = None,
        sa_radii: List[float] = None,
        sa_nsamples: List[int] = None,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.pillar_size = pillar_size
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.W = int(round((self.x_max - self.x_min) / pillar_size))
        self.H = int(round((self.y_max - self.y_min) / pillar_size))
        self.num_pillars = self.H * self.W

        # Augmented per-point features:
        #   original C + (x_c, y_c, z_c) offsets from pillar mean (3)
        #   + (x_p, y_p) offsets from pillar center (2)
        aug_dim = input_channels + 5
        self.pfn_linear = nn.Linear(aug_dim, pillar_feat_dim)
        self.pfn_bn = nn.BatchNorm1d(pillar_feat_dim)

        # 2D backbone on the BEV pseudo-image
        self.backbone = nn.Sequential(
            nn.Conv2d(pillar_feat_dim, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(256, feature_dim),
        )
        # LayerNorm prevents encoder collapse to zero (stops posterior collapse)
        self.out_norm = nn.LayerNorm(feature_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: [B, N, C] where C includes (x, y, z, ...)
        Returns:
            latent: [B, feature_dim]
        """
        B, N, C = points.shape
        device = points.device

        x = points[..., 0]
        y = points[..., 1]

        # Compute integer pillar index per point
        px = ((x - self.x_min) / self.pillar_size).long()
        py = ((y - self.y_min) / self.pillar_size).long()
        in_range = (px >= 0) & (px < self.W) & (py >= 0) & (py < self.H)
        px = px.clamp(0, self.W - 1)
        py = py.clamp(0, self.H - 1)
        pillar_idx = py * self.W + px  # [B, N]

        valid_mask = in_range.float()  # [B, N]

        # Per-pillar XYZ mean (for x_c/y_c/z_c augmentation)
        xyz = points[..., :3]
        pillar_sum = torch.zeros(B, self.num_pillars, 3, device=device)
        pillar_count = torch.zeros(B, self.num_pillars, device=device)
        pillar_sum.scatter_add_(
            1, pillar_idx.unsqueeze(-1).expand(-1, -1, 3),
            xyz * valid_mask.unsqueeze(-1)
        )
        pillar_count.scatter_add_(1, pillar_idx, valid_mask)
        pillar_mean = pillar_sum / pillar_count.clamp(min=1.0).unsqueeze(-1)
        per_point_mean = torch.gather(
            pillar_mean, 1, pillar_idx.unsqueeze(-1).expand(-1, -1, 3)
        )
        offset_from_mean = xyz - per_point_mean  # [B, N, 3]

        # Pillar geometric center (for x_p/y_p augmentation)
        pillar_center_x = self.x_min + (px.float() + 0.5) * self.pillar_size
        pillar_center_y = self.y_min + (py.float() + 0.5) * self.pillar_size
        offset_from_center = torch.stack(
            [x - pillar_center_x, y - pillar_center_y], dim=-1
        )  # [B, N, 2]

        # Augmented per-point features
        aug = torch.cat([points, offset_from_mean, offset_from_center], dim=-1)  # [B, N, C+5]
        aug = aug * valid_mask.unsqueeze(-1)  # zero out invalid points

        # Pillar Feature Net
        F_in = aug.shape[-1]
        feat = self.pfn_linear(aug.reshape(-1, F_in))
        feat = self.pfn_bn(feat)
        feat = F.relu(feat).reshape(B, N, -1)
        feat = feat * valid_mask.unsqueeze(-1)
        D = feat.shape[-1]

        # Scatter max into BEV pillar features
        pillar_feat = torch.full(
            (B, self.num_pillars, D), float("-inf"), device=device
        )
        pillar_feat.scatter_reduce_(
            1, pillar_idx.unsqueeze(-1).expand(-1, -1, D), feat,
            reduce="amax", include_self=True,
        )
        # Empty pillars → 0
        pillar_feat = torch.where(
            torch.isinf(pillar_feat), torch.zeros_like(pillar_feat), pillar_feat
        )

        # BEV pseudo-image: [B, D, H, W]
        bev = pillar_feat.permute(0, 2, 1).reshape(B, D, self.H, self.W)

        # 2D backbone
        bev = self.backbone(bev)  # [B, 256, h, w]

        # Global pool
        global_feat = bev.mean(dim=[2, 3])  # [B, 256]

        latent = self.global_mlp(global_feat)
        latent = self.out_norm(latent)
        return latent
