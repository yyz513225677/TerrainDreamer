"""route_visualizer_node — convert live odometry and recorded JSONL into
nav_msgs/Path topics so RViz / dashboards can draw the route.

Publishes:
  /lunar_jackal/live_path       (nav_msgs/Path)  built from incoming /odom
  /lunar_jackal/recorded_path   (nav_msgs/Path)  loaded once from JSONL,
                                                  if `recorded_jsonl` param set
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional


def load_recorded_path(jsonl_path: Path, frame_id: str = "odom"):
    """Pure-function loader — returns a list of (x, y, z) tuples.

    Importable by tests; does not depend on rclpy.
    """
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"recorded JSONL not found: {jsonl_path}")
    points = []
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = rec.get("pose", {})
            points.append((float(p.get("x", 0.0)),
                           float(p.get("y", 0.0)),
                           float(p.get("z", 0.0))))
    return points


def main(args: list | None = None) -> int:  # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, QoSDurabilityPolicy
        from nav_msgs.msg import Odometry, Path
        from geometry_msgs.msg import PoseStamped
    except ImportError:
        print("[route_visualizer_node] rclpy not available", file=sys.stderr)
        return 1

    rclpy.init(args=args)
    node = Node("route_visualizer_node")

    odom_topic = node.declare_parameter(
        "odom_topic", "/lunar_jackal/odom").value
    frame_id = node.declare_parameter("frame_id", "odom").value
    max_points = int(node.declare_parameter("max_points", 5000).value)
    recorded_jsonl = node.declare_parameter("recorded_jsonl", "").value

    latched = QoSProfile(depth=1)
    latched.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

    live_pub = node.create_publisher(Path, "/lunar_jackal/live_path", 10)
    rec_pub = node.create_publisher(Path, "/lunar_jackal/recorded_path",
                                    latched)

    live_path = Path()
    live_path.header.frame_id = frame_id

    def on_odom(msg):
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = msg.header.stamp
        ps.pose = msg.pose.pose
        live_path.poses.append(ps)
        if len(live_path.poses) > max_points:
            del live_path.poses[: len(live_path.poses) - max_points]
        live_path.header.stamp = msg.header.stamp
        live_pub.publish(live_path)

    node.create_subscription(Odometry, odom_topic, on_odom, 10)

    # Recorded path — publish once (latched) on startup if provided.
    if recorded_jsonl:
        try:
            pts = load_recorded_path(Path(recorded_jsonl), frame_id=frame_id)
            rp = Path()
            rp.header.frame_id = frame_id
            for x, y, z in pts:
                ps = PoseStamped()
                ps.header.frame_id = frame_id
                ps.pose.position.x = x
                ps.pose.position.y = y
                ps.pose.position.z = z
                ps.pose.orientation.w = 1.0
                rp.poses.append(ps)
            rec_pub.publish(rp)
            node.get_logger().info(
                f"recorded path loaded: {len(pts)} points "
                f"from {recorded_jsonl}")
        except (FileNotFoundError, ValueError) as exc:
            node.get_logger().warn(str(exc))

    node.get_logger().info(
        f"route_visualizer_node started; live odom={odom_topic} "
        f"frame={frame_id}")

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
