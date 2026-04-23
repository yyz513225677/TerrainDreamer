"""
Sensor Fusion: LiDAR + IMU → Ego-State Estimation
===================================================
Extended Kalman Filter fusing IMU (high-rate prediction)
with LiDAR-based pose corrections (scan matching or odometry).

State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw]  (9-DoF)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from terrain_dreamer.envs.sensors.velodyne_vlp32 import PointCloud
from terrain_dreamer.envs.sensors.imu_driver import IMUReading


@dataclass
class EgoState:
    """Current vehicle state estimate."""
    timestamp: float = 0.0
    # Position in world frame (meters) — origin = start position
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Velocity in world frame (m/s)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Euler angles (rad): roll, pitch, yaw
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Covariance (9×9)
    covariance: np.ndarray = field(
        default_factory=lambda: np.eye(9) * 0.1
    )

    @property
    def x(self) -> float:
        return self.position[0]

    @property
    def y(self) -> float:
        return self.position[1]

    @property
    def yaw(self) -> float:
        return self.orientation[2]

    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity[:2])

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity, self.orientation])

    @staticmethod
    def from_vector(vec: np.ndarray, timestamp: float = 0.0) -> "EgoState":
        return EgoState(
            timestamp=timestamp,
            position=vec[0:3].copy(),
            velocity=vec[3:6].copy(),
            orientation=vec[6:9].copy(),
        )


class SensorFusion:
    """
    EKF-based sensor fusion.

    Prediction step: IMU accelerometer + gyroscope → dead reckoning
    Update step:     LiDAR-derived odometry → correct drift
    """

    def __init__(
        self,
        Q_position: float = 0.01,
        Q_velocity: float = 0.1,
        Q_orientation: float = 0.001,
        gravity: float = 9.81,
    ):
        self.gravity = gravity

        # State
        self.state = EgoState()
        self._x = np.zeros(9)
        self._P = np.eye(9) * 0.1

        # Process noise (tunable)
        self._Q = np.diag([
            Q_position, Q_position, Q_position,
            Q_velocity, Q_velocity, Q_velocity,
            Q_orientation, Q_orientation, Q_orientation,
        ])

        # LiDAR observation noise (position + yaw from scan matching)
        self._R_lidar = np.diag([0.05, 0.05, 0.1, 0.02])  # x, y, z, yaw

        self._last_imu_time: Optional[float] = None
        self._initialized = False

    def predict_imu(self, reading: IMUReading):
        """
        IMU prediction step.
        Integrate accelerometer and gyroscope to propagate state.
        """
        if self._last_imu_time is None:
            self._last_imu_time = reading.timestamp
            return

        dt = reading.timestamp - self._last_imu_time
        if dt <= 0 or dt > 0.5:
            self._last_imu_time = reading.timestamp
            return
        self._last_imu_time = reading.timestamp

        roll, pitch, yaw = self._x[6:9]

        # Rotation matrix: body → world
        R = self._euler_to_rotation(roll, pitch, yaw)

        # Accelerometer: remove gravity then rotate to world frame
        accel_body = reading.accel.copy()
        accel_body[2] -= self.gravity
        accel_world = R @ accel_body

        # Gyroscope: rotate angular velocity to Euler rates
        gyro = reading.gyro

        # State transition
        # Position += velocity * dt + 0.5 * accel * dt²
        self._x[0:3] += self._x[3:6] * dt + 0.5 * accel_world * dt ** 2
        # Velocity += accel * dt
        self._x[3:6] += accel_world * dt
        # Orientation += gyro * dt (simplified — use proper quaternion for production)
        self._x[6:9] += gyro * dt
        # Wrap yaw to [-π, π]
        self._x[8] = np.arctan2(np.sin(self._x[8]), np.cos(self._x[8]))

        # Jacobian of state transition (linearized)
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        # Propagate covariance
        self._P = F @ self._P @ F.T + self._Q * dt

        self._update_state_output()

    def update_lidar(
        self,
        position: np.ndarray,
        yaw: float,
        covariance: Optional[np.ndarray] = None,
    ):
        """
        LiDAR correction step.
        Accepts a pose estimate from scan matching / ICP / odometry.

        Args:
            position: [x, y, z] in world frame
            yaw: heading angle in radians
            covariance: 4×4 measurement covariance (optional)
        """
        # Observation: z = [x, y, z, yaw]
        z = np.array([position[0], position[1], position[2], yaw])

        # Observation model: H maps state → observation
        H = np.zeros((4, 9))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # z
        H[3, 8] = 1.0  # yaw

        R = covariance if covariance is not None else self._R_lidar

        # Innovation
        y = z - H @ self._x
        y[3] = np.arctan2(np.sin(y[3]), np.cos(y[3]))  # Wrap angle

        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)

        self._x += K @ y
        self._P = (np.eye(9) - K @ H) @ self._P

        self._update_state_output()

    def get_state(self) -> EgoState:
        return self.state

    def reset(self, position: Optional[np.ndarray] = None):
        """Reset filter state (e.g., on GPS fix)."""
        self._x = np.zeros(9)
        if position is not None:
            self._x[0:3] = position
        self._P = np.eye(9) * 0.1
        self._last_imu_time = None
        self._update_state_output()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_state_output(self):
        import time as _time
        self.state = EgoState(
            timestamp=_time.time(),
            position=self._x[0:3].copy(),
            velocity=self._x[3:6].copy(),
            orientation=self._x[6:9].copy(),
            covariance=self._P.copy(),
        )

    @staticmethod
    def _euler_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """ZYX Euler angles → 3×3 rotation matrix."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [  -sp,           cp*sr,           cp*cr     ],
        ])
        return R
