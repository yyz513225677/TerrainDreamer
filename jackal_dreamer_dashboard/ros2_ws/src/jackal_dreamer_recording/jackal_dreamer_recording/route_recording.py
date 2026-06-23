"""route_recording_node — captures Jackal telemetry as JSONL demos.

Subscribes (control):
  /dashboard/recording/start (std_msgs/String, JSON body)
  /dashboard/recording/stop  (std_msgs/Empty)
  /dashboard/recording/save  (std_msgs/String, route_id payload)

Subscribes (telemetry):
  /scan, /imu/data, /ground_truth/odom, /dreamer/action,
  /collision_status

Writes:
  data/demonstrations/<route_id>.jsonl
  data/demonstrations/<route_id>_lidar/<frame_idx>.npy
  data/routes/<route_id>.json
  data/replay_buffer/index.json (appended)
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry

from jackal_dreamer_dashboard_msgs.msg import CollisionStatus

from .schema import validate_sample


DEFAULT_DATA_ROOT = ("/home/rickslab3/Documents/Leo/terrain_dreamer/"
                     "jackal_dreamer_dashboard/data")
RECORD_HZ = 10.0


def _quat_to_rpy(x: float, y: float, z: float, w: float
                 ) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1 \
        else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class RouteRecordingNode(Node):
    def __init__(self) -> None:
        super().__init__("route_recording_node")

        self.declare_parameter("data_root", DEFAULT_DATA_ROOT)
        self._data_root = Path(
            self.get_parameter("data_root").get_parameter_value().string_value
            or DEFAULT_DATA_ROOT)
        for sub in ("demonstrations", "routes", "replay_buffer"):
            (self._data_root / sub).mkdir(parents=True, exist_ok=True)

        # recording state
        self._active: bool = False
        self._route_id: Optional[str] = None
        self._operator_id: str = ""
        self._environment_id: str = ""
        self._jsonl_path: Optional[Path] = None
        self._lidar_dir: Optional[Path] = None
        self._jsonl_fh = None
        self._frame_idx: int = 0
        self._t_start: float = 0.0

        # latest telemetry
        self._latest_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self._latest_odom = {"linear_x": 0.0, "angular_z": 0.0}
        self._latest_imu = {
            "rpy": [0.0, 0.0, 0.0],
            "angvel": [0.0, 0.0, 0.0],
            "linacc": [0.0, 0.0, 0.0],
        }
        self._latest_action = {"linear_x": 0.0, "angular_z": 0.0}
        self._latest_collision = False
        self._latest_lidar: Optional[np.ndarray] = None

        # subscribers
        self.create_subscription(String, "/dashboard/recording/start",
                                 self._on_start, 10)
        self.create_subscription(Empty, "/dashboard/recording/stop",
                                 self._on_stop, 10)
        self.create_subscription(String, "/dashboard/recording/save",
                                 self._on_save, 10)

        self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
        self.create_subscription(Imu, "/imu/data", self._on_imu, 10)
        self.create_subscription(Odometry, "/ground_truth/odom",
                                 self._on_odom, 10)
        self.create_subscription(Twist, "/dreamer/action",
                                 self._on_action, 10)
        self.create_subscription(CollisionStatus, "/collision_status",
                                 self._on_collision, 10)

        self.timer = self.create_timer(1.0 / RECORD_HZ, self._tick)
        self.get_logger().info(
            f"route_recording_node started; data_root={self._data_root}")

    # ----------------------------------------------------------------
    # control callbacks
    # ----------------------------------------------------------------
    def _on_start(self, msg: String) -> None:
        try:
            body = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            self.get_logger().warn(f"Invalid start payload: {msg.data!r}")
            body = {}
        route_id = (body.get("route_id")
                    or f"demo_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}")
        self._operator_id = str(body.get("operator_id", ""))
        self._environment_id = str(body.get("environment_id", ""))
        self._begin_recording(route_id)

    def _on_stop(self, _msg: Empty) -> None:
        self._end_recording()

    def _on_save(self, msg: String) -> None:
        rid = msg.data.strip() or self._route_id
        if not rid:
            self.get_logger().warn("Save requested with no active route.")
            return
        # If still active, close the JSONL first.
        if self._active:
            self._end_recording()
        self._write_manifest_and_index(rid)

    # ----------------------------------------------------------------
    # telemetry callbacks
    # ----------------------------------------------------------------
    def _on_scan(self, msg: LaserScan) -> None:
        arr = np.asarray(msg.ranges, dtype=np.float32)
        # Replace inf/nan with 0 for compactness.
        arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
        self._latest_lidar = arr

    def _on_imu(self, msg: Imu) -> None:
        q = msg.orientation
        roll, pitch, yaw = _quat_to_rpy(q.x, q.y, q.z, q.w)
        self._latest_imu = {
            "rpy": [roll, pitch, yaw],
            "angvel": [msg.angular_velocity.x,
                       msg.angular_velocity.y,
                       msg.angular_velocity.z],
            "linacc": [msg.linear_acceleration.x,
                       msg.linear_acceleration.y,
                       msg.linear_acceleration.z],
        }

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = _quat_to_rpy(q.x, q.y, q.z, q.w)
        self._latest_pose = {"x": float(p.x), "y": float(p.y),
                             "yaw": float(yaw)}
        self._latest_odom = {
            "linear_x": float(msg.twist.twist.linear.x),
            "angular_z": float(msg.twist.twist.angular.z),
        }

    def _on_action(self, msg: Twist) -> None:
        self._latest_action = {"linear_x": float(msg.linear.x),
                               "angular_z": float(msg.angular.z)}

    def _on_collision(self, msg: CollisionStatus) -> None:
        self._latest_collision = bool(msg.in_contact)

    # ----------------------------------------------------------------
    # recording machinery
    # ----------------------------------------------------------------
    def _begin_recording(self, route_id: str) -> None:
        if self._active:
            self.get_logger().warn(
                f"Recording already active ({self._route_id}); ignoring start.")
            return
        self._route_id = route_id
        self._jsonl_path = self._data_root / "demonstrations" / f"{route_id}.jsonl"
        self._lidar_dir = self._data_root / "demonstrations" / f"{route_id}_lidar"
        self._lidar_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_fh = open(self._jsonl_path, "w", encoding="utf-8")
        self._frame_idx = 0
        self._t_start = time.time()
        self._active = True
        self.get_logger().info(
            f"Recording started: {route_id} -> {self._jsonl_path}")

    def _end_recording(self) -> None:
        if not self._active:
            return
        try:
            if self._jsonl_fh is not None:
                self._jsonl_fh.flush()
                self._jsonl_fh.close()
        finally:
            self._jsonl_fh = None
            self._active = False
        self.get_logger().info(
            f"Recording stopped: {self._route_id} "
            f"({self._frame_idx} frames)")

    def _tick(self) -> None:
        if not self._active or self._jsonl_fh is None:
            return

        # LiDAR side-car
        lidar_ref = ""
        if self._latest_lidar is not None and self._lidar_dir is not None:
            fname = f"{self._frame_idx:06d}.npy"
            np.save(self._lidar_dir / fname, self._latest_lidar)
            lidar_ref = str(
                (Path("data/demonstrations")
                 / f"{self._route_id}_lidar" / fname))

        sample = {
            "timestamp":    time.time(),
            "route_id":     self._route_id,
            "operator_id":  self._operator_id,
            "environment_id": self._environment_id,
            "pose":         self._latest_pose,
            "odom":         self._latest_odom,
            "imu": {
                "roll":  self._latest_imu["rpy"][0],
                "pitch": self._latest_imu["rpy"][1],
                "yaw":   self._latest_imu["rpy"][2],
                "angular_velocity": {
                    "x": self._latest_imu["angvel"][0],
                    "y": self._latest_imu["angvel"][1],
                    "z": self._latest_imu["angvel"][2],
                },
                "linear_acceleration": {
                    "x": self._latest_imu["linacc"][0],
                    "y": self._latest_imu["linacc"][1],
                    "z": self._latest_imu["linacc"][2],
                },
            },
            "lidar_ref":   lidar_ref,
            "action":      self._latest_action,
            "collision":   self._latest_collision,
            "reward_est":  0.0,
            "priority":    1.0,
        }
        errs = validate_sample(sample)
        if errs:
            self.get_logger().warn(
                f"Schema validation errors at frame {self._frame_idx}: {errs}")

        self._jsonl_fh.write(json.dumps(sample) + "\n")
        self._jsonl_fh.flush()
        self._frame_idx += 1

    # ----------------------------------------------------------------
    # manifest + index
    # ----------------------------------------------------------------
    def _write_manifest_and_index(self, route_id: str) -> None:
        samples_path = (Path("data/demonstrations") / f"{route_id}.jsonl")
        abs_samples_path = self._data_root / "demonstrations" / f"{route_id}.jsonl"
        abs_lidar_dir = self._data_root / "demonstrations" / f"{route_id}_lidar"

        frame_count = 0
        if abs_samples_path.exists():
            with abs_samples_path.open("r", encoding="utf-8") as fh:
                frame_count = sum(1 for _ in fh)

        duration_s = (
            float(frame_count) / RECORD_HZ if frame_count else 0.0)
        manifest = {
            "route_id":       route_id,
            "operator_id":    self._operator_id,
            "environment_id": self._environment_id,
            "started_at":     datetime.now(timezone.utc).isoformat(),
            "duration_s":     duration_s,
            "frame_count":    frame_count,
            "priority":       1.0,
            "samples_path":   str(samples_path),
            "lidar_dir":      str(Path("data/demonstrations") /
                                   f"{route_id}_lidar"),
        }
        manifest_path = self._data_root / "routes" / f"{route_id}.json"
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        self.get_logger().info(f"Wrote manifest: {manifest_path}")

        index_path = self._data_root / "replay_buffer" / "index.json"
        if index_path.exists():
            try:
                with index_path.open("r", encoding="utf-8") as fh:
                    index = json.load(fh)
                if not isinstance(index, list):
                    index = []
            except json.JSONDecodeError:
                index = []
        else:
            index = []

        index.append({
            "route_id":       route_id,
            "priority":       1.0,
            "operator_id":    self._operator_id,
            "environment_id": self._environment_id,
            "samples_path":   str(samples_path),
            "frame_count":    frame_count,
            "duration_s":     duration_s,
        })
        with index_path.open("w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2)
        self.get_logger().info(f"Appended index entry to {index_path}")
        _ = abs_lidar_dir  # silence unused (kept for future use)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RouteRecordingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # If we were recording, close the file.
        if node._active:
            node._end_recording()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
