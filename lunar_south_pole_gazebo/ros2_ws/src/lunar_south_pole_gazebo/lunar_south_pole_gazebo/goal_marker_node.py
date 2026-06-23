"""goal_marker_node — publish destination beacons + labels as
visualization_msgs/MarkerArray, so RViz (and any dashboard) renders
them with zero extra code.

Goals are loaded from config/lunar_goals.yaml (path passed via
`goals_yaml` parameter or via the LUNAR_GOALS_YAML env var).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _resolve_goals_yaml_default() -> str:
    """Best-effort default — installed share/<pkg>/config/lunar_goals.yaml."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory("lunar_south_pole_gazebo"))
        candidate = share / "config" / "lunar_goals.yaml"
        if candidate.is_file():
            return str(candidate)
    except Exception:
        pass
    return os.environ.get("LUNAR_GOALS_YAML",
                          "config/lunar_goals.yaml")


def load_goals(path: Path) -> tuple[list[dict], dict]:
    """Pure-function loader — importable by tests."""
    import yaml
    if not path.is_file():
        raise FileNotFoundError(f"goals yaml not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    goals = data.get("goals", [])
    if not isinstance(goals, list) or not goals:
        raise ValueError(f"{path} has no 'goals:' list")
    required = {"goal_id", "name", "x", "y"}
    for i, g in enumerate(goals):
        missing = required - set(g.keys())
        if missing:
            raise ValueError(f"goal #{i}: missing fields {sorted(missing)}")
    defaults = data.get("defaults", {})
    return goals, defaults


def main(args: list | None = None) -> int:  # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, QoSDurabilityPolicy
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA, Header
    except ImportError:
        print("[goal_marker_node] rclpy / visualization_msgs not "
              "available — is ROS 2 sourced?", file=sys.stderr)
        return 1

    rclpy.init(args=args)
    node = Node("goal_marker_node")
    goals_yaml = node.declare_parameter(
        "goals_yaml", _resolve_goals_yaml_default()).value
    frame_id = node.declare_parameter("frame_id", "odom").value
    publish_rate_hz = float(node.declare_parameter(
        "publish_rate_hz", 1.0).value)

    try:
        goals, defaults = load_goals(Path(goals_yaml))
    except (FileNotFoundError, ValueError) as exc:
        node.get_logger().error(str(exc))
        # publish empty MarkerArray so downstream consumers see "no goals"
        goals, defaults = [], {}

    qos = QoSProfile(depth=1)
    qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
    pub = node.create_publisher(MarkerArray, "/lunar_jackal/goals", qos)

    beacon_radius_m = float(defaults.get("beacon_radius_m", 0.4))
    beacon_height_m = float(defaults.get("beacon_height_m", 25.0))
    text_height_m = float(defaults.get("text_height_m", 6.0))
    default_arrival_radius_m = float(defaults.get("arrival_radius_m", 3.0))

    def make_markers() -> MarkerArray:
        ma = MarkerArray()
        for i, g in enumerate(goals):
            x = float(g["x"]); y = float(g["y"]); z = float(g.get("z", 0.0))
            color = g.get("color", [1.0, 0.85, 0.10, 1.0])
            c = ColorRGBA(r=float(color[0]), g=float(color[1]),
                          b=float(color[2]), a=float(color[3])
                          if len(color) > 3 else 1.0)

            # Beacon cylinder
            beacon = Marker()
            beacon.header.frame_id = frame_id
            beacon.ns = "beacon"
            beacon.id = i
            beacon.type = Marker.CYLINDER
            beacon.action = Marker.ADD
            beacon.pose.position.x = x
            beacon.pose.position.y = y
            beacon.pose.position.z = z + beacon_height_m / 2.0
            beacon.pose.orientation.w = 1.0
            beacon.scale.x = beacon_radius_m * 2.0
            beacon.scale.y = beacon_radius_m * 2.0
            beacon.scale.z = beacon_height_m
            beacon.color = c
            beacon.lifetime.sec = 0
            ma.markers.append(beacon)

            # Arrival-radius ring (flat cylinder)
            radius = float(g.get("arrival_radius_m",
                                 default_arrival_radius_m))
            ring = Marker()
            ring.header.frame_id = frame_id
            ring.ns = "arrival"
            ring.id = i
            ring.type = Marker.CYLINDER
            ring.action = Marker.ADD
            ring.pose.position.x = x
            ring.pose.position.y = y
            ring.pose.position.z = z + 0.05
            ring.pose.orientation.w = 1.0
            ring.scale.x = radius * 2.0
            ring.scale.y = radius * 2.0
            ring.scale.z = 0.1
            ring.color = ColorRGBA(r=c.r, g=c.g, b=c.b, a=0.30)
            ma.markers.append(ring)

            # Text label
            text = Marker()
            text.header.frame_id = frame_id
            text.ns = "label"
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = z + beacon_height_m + 2.0
            text.pose.orientation.w = 1.0
            text.scale.z = text_height_m
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text.text = f"{g['goal_id']} — {g['name']}"
            ma.markers.append(text)
        return ma

    # MarkerArray is published ONCE on startup. Late-joining
    # subscribers receive it via the TRANSIENT_LOCAL durability set
    # on the publisher above. Re-publication is unnecessary unless
    # the YAML changes — which requires a node restart anyway.
    pub.publish(make_markers())
    node.get_logger().info(
        f"goal_marker_node published {len(goals)} goals on "
        f"/lunar_jackal/goals (frame={frame_id}, latched)")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
