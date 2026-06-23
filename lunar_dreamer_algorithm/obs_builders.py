"""Pure-numpy helpers that convert raw sensor data into the algorithm's
observation tensors. No ROS imports here — the ROS-side script reads its
own message types and passes plain numpy arrays in.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch


# ────────────────────────────────────────────────────────────────────────────
# LiDAR point cloud  →  BEV [4, H, W]
# ────────────────────────────────────────────────────────────────────────────
def points_to_bev(
    points: np.ndarray,
    grid_size: int = 128,
    extent_m: float = 15.0,
    height_floor_m: float = -0.5,
    height_ceiling_m: float = 4.0,
) -> np.ndarray:
    """Rasterize an Nx3 lidar point cloud (rover frame) into a BEV tensor.

    Output shape: (4, grid_size, grid_size) on float32.
    Channel layout:
        0: occupancy   — 1 where any non-ground point falls
        1: max_height  — highest z in the cell, normalized to [0, 1]
        2: elevation   — lowest z in the cell (terrain surface), [0, 1]
        3: roughness   — (max - min) z in the cell, [0, 1]

    Points must be in the rover's body frame (+X forward, +Y left). Points
    outside the (±extent_m, ±extent_m) window are discarded.
    """
    if points.size == 0:
        return np.zeros((4, grid_size, grid_size), dtype=np.float32)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mask = (
        (x >= -extent_m) & (x < extent_m) &
        (y >= -extent_m) & (y < extent_m) &
        (z >= height_floor_m) & (z < height_ceiling_m)
    )
    x, y, z = x[mask], y[mask], z[mask]
    if x.size == 0:
        return np.zeros((4, grid_size, grid_size), dtype=np.float32)

    # World-frame (+X fwd, +Y left)  →  image (row=forward grows up, col=left grows right)
    cell = (2 * extent_m) / grid_size
    col = ((y + extent_m) / cell).astype(np.int32)
    row = ((extent_m - x) / cell).astype(np.int32)
    col = np.clip(col, 0, grid_size - 1)
    row = np.clip(row, 0, grid_size - 1)
    flat = row * grid_size + col

    bev = np.zeros((4, grid_size, grid_size), dtype=np.float32)
    occ = np.zeros(grid_size * grid_size, dtype=np.float32)
    max_h = np.full(grid_size * grid_size, -np.inf, dtype=np.float32)
    min_h = np.full(grid_size * grid_size, np.inf, dtype=np.float32)

    # Vectorized per-cell aggregation using np.maximum.at / np.minimum.at.
    occ[flat] = 1.0
    np.maximum.at(max_h, flat, z)
    np.minimum.at(min_h, flat, z)

    span = height_ceiling_m - height_floor_m
    norm = lambda v: np.clip((v - height_floor_m) / span, 0.0, 1.0)

    bev[0] = occ.reshape(grid_size, grid_size)
    # Replace -inf / +inf in untouched cells with 0 (then norm clips to 0).
    max_h = np.where(np.isfinite(max_h), max_h, height_floor_m)
    min_h = np.where(np.isfinite(min_h), min_h, height_floor_m)
    bev[1] = norm(max_h).reshape(grid_size, grid_size)
    bev[2] = norm(min_h).reshape(grid_size, grid_size)
    bev[3] = np.clip((max_h - min_h) / span, 0.0, 1.0).reshape(grid_size, grid_size)
    return bev


# ────────────────────────────────────────────────────────────────────────────
# IMU msg  →  [12]
# ────────────────────────────────────────────────────────────────────────────
def quat_to_rpy(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def build_imu_vec(
    lin_acc: Tuple[float, float, float],
    ang_vel: Tuple[float, float, float],
    rpy: Tuple[float, float, float],
    body_vel: Tuple[float, float, float],
) -> np.ndarray:
    """Pack a 12-vector: lin_acc(3) + ang_vel(3) + rpy(3) + body_vel(3)."""
    return np.array([
        *lin_acc, *ang_vel, *rpy, *body_vel,
    ], dtype=np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Goal vec — relative target in rover frame
# ────────────────────────────────────────────────────────────────────────────
def build_goal_vec(
    rover_xy: Tuple[float, float],
    yaw: float,
    target_xy: Tuple[float, float],
) -> np.ndarray:
    dx_world = target_xy[0] - rover_xy[0]
    dy_world = target_xy[1] - rover_xy[1]
    distance = math.hypot(dx_world, dy_world)
    bearing_world = math.atan2(dy_world, dx_world)
    # Rotate world delta into the rover frame so the policy sees relative coords.
    c, s = math.cos(-yaw), math.sin(-yaw)
    dx_body = c * dx_world - s * dy_world
    dy_body = s * dx_world + c * dy_world
    bearing_rel = math.atan2(math.sin(bearing_world - yaw),
                             math.cos(bearing_world - yaw))
    return np.array([dx_body, dy_body, distance, bearing_rel], dtype=np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Pack into a torch obs dict
# ────────────────────────────────────────────────────────────────────────────
def pack_obs(
    bev: np.ndarray,
    imu_vec: np.ndarray,
    goal_vec: np.ndarray,
    phase_one_hot: Tuple[float, float],
    prev_action: np.ndarray,
) -> dict:
    return {
        "bev":   torch.from_numpy(bev),
        "imu":   torch.from_numpy(imu_vec),
        "goal":  torch.from_numpy(goal_vec),
        "phase": torch.tensor(phase_one_hot, dtype=torch.float32),
        "prev_action": torch.from_numpy(prev_action.astype(np.float32)),
    }
