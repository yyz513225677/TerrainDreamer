"""
Point Cloud Processor
======================
Transforms raw VLP-32C scans into clean, structured tensors
ready for the terrain encoder.

Pipeline:
  Raw scan → ROI crop → Voxel downsample → Ground segmentation
           → Normal estimation → Tensor output
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from terrain_dreamer.envs.sensors.velodyne_vlp32 import PointCloud


@dataclass
class ProcessedCloud:
    """Processed point cloud ready for neural network input."""
    timestamp: float
    # Core data [N, C] where C = (x, y, z, intensity, nx, ny, nz, height_above_ground)
    features: np.ndarray
    # Separate arrays for convenience
    xyz: np.ndarray             # [N, 3]
    normals: np.ndarray         # [N, 3]
    intensity: np.ndarray       # [N]
    height_above_ground: np.ndarray  # [N]
    # Ground mask
    is_ground: np.ndarray       # [N] boolean
    # Number of points
    num_points: int


class PointCloudProcessor:
    """Process raw LiDAR scans into features for the world model."""

    def __init__(
        self,
        voxel_size: float = 0.1,
        roi_x: Tuple[float, float] = (-5.0, 50.0),
        roi_y: Tuple[float, float] = (-25.0, 25.0),
        roi_z: Tuple[float, float] = (-2.0, 3.0),
        ground_threshold: float = 0.3,
        ransac_iterations: int = 20,
        max_points: int = 32768,
    ):
        self.voxel_size = voxel_size
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_z = roi_z
        self.ground_threshold = ground_threshold
        self.ransac_iterations = ransac_iterations
        self.max_points = max_points

    def process(self, scan: PointCloud) -> ProcessedCloud:
        """Full processing pipeline."""
        points = scan.points.copy()  # [N, 4] (x, y, z, intensity)

        # 1. Region of interest crop
        points = self._crop_roi(points)

        # 2. Voxel downsampling
        points = self._voxel_downsample(points)

        # Empty after cropping — return an empty ProcessedCloud so the caller
        # can pad to its own N. Happens e.g. on the very first sim tick when
        # the LiDAR hasn't produced a scan yet.
        if points.shape[0] == 0:
            return ProcessedCloud(
                timestamp=scan.timestamp,
                features=np.zeros((0, 8), dtype=np.float32),
                xyz=np.zeros((0, 3), dtype=np.float32),
                normals=np.zeros((0, 3), dtype=np.float32),
                intensity=np.zeros((0,), dtype=np.float32),
                height_above_ground=np.zeros((0,), dtype=np.float32),
                is_ground=np.zeros((0,), dtype=bool),
                num_points=0,
            )

        # 3. Ground segmentation via RANSAC
        is_ground, ground_plane = self._segment_ground(points[:, :3])

        # 4. Height above ground
        if ground_plane is not None:
            a, b, c, d = ground_plane
            heights = (a * points[:, 0] + b * points[:, 1]
                       + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
        else:
            heights = points[:, 2] - np.min(points[:, 2])

        # 5. Surface normals (local PCA)
        normals = self._estimate_normals(points[:, :3], k=10)

        # 6. Limit point count (random subsample if needed)
        if len(points) > self.max_points:
            idx = np.random.choice(len(points), self.max_points, replace=False)
            points = points[idx]
            is_ground = is_ground[idx]
            heights = heights[idx]
            normals = normals[idx]

        # 7. Assemble feature tensor
        features = np.concatenate([
            points[:, :3],                    # x, y, z
            points[:, 3:4],                   # intensity
            normals,                          # nx, ny, nz
            heights.reshape(-1, 1),           # height above ground
        ], axis=1)  # [N, 8]

        return ProcessedCloud(
            timestamp=scan.timestamp,
            features=features.astype(np.float32),
            xyz=points[:, :3].astype(np.float32),
            normals=normals.astype(np.float32),
            intensity=points[:, 3].astype(np.float32),
            height_above_ground=heights.astype(np.float32),
            is_ground=is_ground,
            num_points=len(points),
        )

    # ------------------------------------------------------------------
    # Processing Steps
    # ------------------------------------------------------------------

    def _crop_roi(self, points: np.ndarray) -> np.ndarray:
        """Keep only points within the region of interest."""
        mask = (
            (points[:, 0] >= self.roi_x[0]) & (points[:, 0] <= self.roi_x[1]) &
            (points[:, 1] >= self.roi_y[0]) & (points[:, 1] <= self.roi_y[1]) &
            (points[:, 2] >= self.roi_z[0]) & (points[:, 2] <= self.roi_z[1])
        )
        return points[mask]

    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """Hash-based voxel downsampling (fast, no Open3D dependency)."""
        if self.voxel_size <= 0:
            return points

        # Quantize to voxel grid
        coords = np.floor(points[:, :3] / self.voxel_size).astype(np.int32)

        # Unique voxel keys
        _, unique_idx = np.unique(
            coords, axis=0, return_index=True
        )
        return points[np.sort(unique_idx)]

    def _segment_ground(
        self, xyz: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        RANSAC ground plane segmentation, vectorized over all iterations.
        Returns (is_ground mask, plane coefficients [a, b, c, d]).
        """
        n = len(xyz)
        if n < 10:
            return np.zeros(n, dtype=bool), None

        iters = self.ransac_iterations
        # Sample all triplets at once [iters, 3]
        samples = np.stack([
            np.random.choice(n, 3, replace=False) for _ in range(iters)
        ])  # [iters, 3]

        p1 = xyz[samples[:, 0]]  # [iters, 3]
        p2 = xyz[samples[:, 1]]
        p3 = xyz[samples[:, 2]]

        v1 = p2 - p1  # [iters, 3]
        v2 = p3 - p1
        normals = np.cross(v1, v2)  # [iters, 3]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)  # [iters, 1]

        valid = (norms[:, 0] > 1e-8)
        if not valid.any():
            return np.zeros(n, dtype=bool), None

        normals = normals[valid] / norms[valid]  # [k, 3]
        p1 = p1[valid]                           # [k, 3]
        ds = -np.einsum('ki,ki->k', normals, p1) # [k]

        # Distances of all N points to all k planes: [k, N]
        dists = np.abs(xyz @ normals.T + ds[np.newaxis, :])  # [N, k]
        inlier_counts = (dists < self.ground_threshold).sum(axis=0)  # [k]

        best_k = int(np.argmax(inlier_counts))
        best_normal = normals[best_k]
        best_d = ds[best_k]
        best_inliers = dists[:, best_k] < self.ground_threshold
        best_plane = np.array([best_normal[0], best_normal[1], best_normal[2], best_d])

        return best_inliers, best_plane

    def _estimate_normals(self, xyz: np.ndarray, k: int = 10) -> np.ndarray:
        """Estimate point normals via local PCA (k-nearest neighbors), vectorized."""
        from scipy.spatial import cKDTree

        n = len(xyz)
        normals = np.zeros((n, 3), dtype=np.float32)

        if n < k:
            normals[:, 2] = 1.0  # Default: up
            return normals

        tree = cKDTree(xyz)
        _, indices = tree.query(xyz, k=k)  # [n, k]

        # Vectorized PCA: compute covariance for all points simultaneously
        neighbors = xyz[indices]                          # [n, k, 3]
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)  # [n, k, 3]
        covs = np.einsum('nki,nkj->nij', centered, centered) / k     # [n, 3, 3]
        _, eigvecs = np.linalg.eigh(covs)                # eigvecs [n, 3, 3]
        normals = eigvecs[:, :, 0].astype(np.float32)    # smallest eigenvector [n, 3]

        # Orient normals upward (z > 0)
        flip = normals[:, 2] < 0
        normals[flip] *= -1

        return normals
