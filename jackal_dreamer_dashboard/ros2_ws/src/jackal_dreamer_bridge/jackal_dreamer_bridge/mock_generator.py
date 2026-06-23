"""Synthetic telemetry generator used when --mock is passed to the bridge.

Generates:
  - figure-8 pose
  - sinusoidal IMU
  - circular LiDAR with one moving obstacle
  - slowly-changing Dreamer losses
  - occasional mode/route events

No ROS dependencies — pure Python.
"""

from __future__ import annotations

import math
import random
import time
from typing import Iterator


N_LIDAR_RAYS = 180
MAX_RANGE = 10.0


def _figure_eight(t: float) -> tuple[float, float, float]:
    # Lissajous figure-eight; period ~ 20 s
    a = 8.0
    b = 4.0
    omega = 2.0 * math.pi / 20.0
    x = a * math.sin(omega * t)
    y = b * math.sin(2.0 * omega * t)
    # tangent yaw
    dx = a * omega * math.cos(omega * t)
    dy = b * 2.0 * omega * math.cos(2.0 * omega * t)
    yaw = math.atan2(dy, dx)
    return x, y, yaw


def _lidar(t: float) -> list[float]:
    # Circular wall at ~6 m + one moving obstacle at ~2 m
    ranges = [6.0 + 0.4 * math.sin(0.7 * t + i * 0.05)
              for i in range(N_LIDAR_RAYS)]
    obstacle_theta = (t * 0.3) % (2 * math.pi)
    obstacle_idx = int(obstacle_theta / (2 * math.pi) * N_LIDAR_RAYS)
    for k in range(-3, 4):
        idx = (obstacle_idx + k) % N_LIDAR_RAYS
        ranges[idx] = 2.0 + 0.05 * abs(k)
    return ranges


def _imu(t: float) -> dict:
    return {
        "rpy": [0.02 * math.sin(t),
                0.03 * math.sin(0.7 * t),
                _figure_eight(t)[2]],
        "angvel": [0.0, 0.0, 0.4 * math.cos(t * 0.5)],
        "linacc": [0.05 * math.sin(2 * t),
                   0.05 * math.cos(2 * t),
                   -9.81],
    }


def _dreamer(t: float) -> dict:
    base = math.exp(-t * 0.005)
    return {
        "wm_loss":          0.6 * base + 0.05 * random.random(),
        "actor_loss":      -0.2 + 0.05 * math.sin(t * 0.1),
        "critic_loss":      0.4 * base + 0.03 * random.random(),
        "bc_loss":          0.3 * base + 0.02 * random.random(),
        "buffer_size":      int(1000 + t * 5),
        "human_ratio":      0.15 + 0.05 * math.sin(t * 0.05),
        "reward_est":       0.5 + 0.3 * math.sin(t * 0.2),
        "value_est":        1.2 + 0.4 * math.sin(t * 0.15),
        "uncertainty_est":  0.1 + 0.05 * math.cos(t * 0.3),
        "action_linear":    0.5 + 0.2 * math.sin(t * 0.4),
        "action_angular":   0.3 * math.cos(t * 0.6),
    }


def telemetry_stream(rate_hz: float = 20.0) -> Iterator[dict]:
    """Yield envelopes shaped like the bridge's WS protocol.

    Each call to next() advances real time by 1/rate_hz seconds via sleep
    only if used directly. Async wrappers should manage their own sleep.
    """
    t0 = time.monotonic()
    tick = 0
    period = 1.0 / rate_hz
    while True:
        t = time.monotonic() - t0
        x, y, yaw = _figure_eight(t)
        imu = _imu(t)
        # telemetry every tick
        yield {
            "topic": "telemetry",
            "payload": {
                "stamp": t,
                "pose": {"x": x, "y": y, "yaw": yaw},
                "odom": {"linear_x": 0.6, "angular_z": 0.2 * math.sin(t)},
                "imu_rpy": imu["rpy"],
                "imu_angvel": imu["angvel"],
                "imu_linacc": imu["linacc"],
                "collision": (tick % 200 == 0 and tick != 0),
                "stuck": max(0.0, 0.3 * math.sin(t * 0.07)),
            },
        }
        # lidar at ~10 Hz
        if tick % max(1, int(rate_hz // 10)) == 0:
            yield {
                "topic": "lidar",
                "payload": {
                    "ranges": _lidar(t),
                    "angle_min": -math.pi,
                    "angle_max": math.pi,
                },
            }
        # dreamer at ~5 Hz
        if tick % max(1, int(rate_hz // 5)) == 0:
            yield {"topic": "dreamer", "payload": _dreamer(t)}
        # mode heartbeat every 1 s
        if tick % max(1, int(rate_hz)) == 0:
            yield {
                "topic": "mode",
                "payload": {"mode": "idle", "estop": False},
            }
        tick += 1
        # Caller may or may not sleep; we leave that to the async loop.
        # Sleeping here would block asyncio.
        _ = period  # signal we know the cadence target
