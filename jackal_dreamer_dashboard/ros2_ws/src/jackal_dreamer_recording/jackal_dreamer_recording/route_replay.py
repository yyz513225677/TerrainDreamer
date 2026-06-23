"""route_replay_node — replays a recorded JSONL demo at the original rate.

Subscribes:
  /dashboard/replay/start (std_msgs/String — route_id)
  /dashboard/replay/stop  (std_msgs/Empty)

Publishes:
  /cmd_vel                  (geometry_msgs/Twist) at 10 Hz
  /dashboard/replay/pose    (geometry_msgs/Pose2D) at 10 Hz
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Empty
from geometry_msgs.msg import Twist, Pose2D


DEFAULT_DATA_ROOT = ("/home/rickslab3/Documents/Leo/terrain_dreamer/"
                     "jackal_dreamer_dashboard/data")
REPLAY_HZ = 10.0


class RouteReplayNode(Node):
    def __init__(self) -> None:
        super().__init__("route_replay_node")

        self.declare_parameter("data_root", DEFAULT_DATA_ROOT)
        self._data_root = Path(
            self.get_parameter("data_root").get_parameter_value().string_value
            or DEFAULT_DATA_ROOT)

        self._samples: list[dict] = []
        self._cursor: int = 0
        self._active: bool = False

        self.create_subscription(String, "/dashboard/replay/start",
                                 self._on_start, 10)
        self.create_subscription(Empty, "/dashboard/replay/stop",
                                 self._on_stop, 10)

        self._pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self._pub_pose = self.create_publisher(
            Pose2D, "/dashboard/replay/pose", 10)

        self.timer = self.create_timer(1.0 / REPLAY_HZ, self._tick)
        self.get_logger().info(
            f"route_replay_node started; data_root={self._data_root}")

    def _on_start(self, msg: String) -> None:
        route_id = (msg.data or "").strip()
        if not route_id:
            self.get_logger().warn("Replay start with empty route_id.")
            return
        path = self._data_root / "demonstrations" / f"{route_id}.jsonl"
        if not path.exists():
            self.get_logger().error(f"Route file not found: {path}")
            return
        loaded: list[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    loaded.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not loaded:
            self.get_logger().error(f"Route {route_id} contains 0 samples.")
            return
        self._samples = loaded
        self._cursor = 0
        self._active = True
        self.get_logger().info(
            f"Replay started: {route_id} ({len(loaded)} samples)")

    def _on_stop(self, _msg: Empty) -> None:
        if self._active:
            self.get_logger().info("Replay stopped by request.")
        self._active = False
        self._samples = []
        self._cursor = 0
        # Final zero-twist for safety.
        self._pub_cmd.publish(Twist())

    def _tick(self) -> None:
        if not self._active:
            return
        if self._cursor >= len(self._samples):
            self.get_logger().info("Replay exhausted.")
            self._active = False
            self._pub_cmd.publish(Twist())
            return

        s = self._samples[self._cursor]
        self._cursor += 1

        action = s.get("action", {})
        tw = Twist()
        tw.linear.x = float(action.get("linear_x", 0.0))
        tw.angular.z = float(action.get("angular_z", 0.0))
        self._pub_cmd.publish(tw)

        pose = s.get("pose", {})
        p2 = Pose2D()
        p2.x = float(pose.get("x", 0.0))
        p2.y = float(pose.get("y", 0.0))
        p2.theta = float(pose.get("yaw", 0.0))
        self._pub_pose.publish(p2)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RouteReplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
