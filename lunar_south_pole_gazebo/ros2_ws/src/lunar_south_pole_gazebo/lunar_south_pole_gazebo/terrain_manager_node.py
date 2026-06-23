"""terrain_manager_node — listens on /dashboard/reset and republishes the
current terrain identifier on /terrain/active.

This node does not regenerate terrain — it only tracks which DEM tile
is loaded. World swapping is a Gazebo-side concern handled by the
launch system.
"""
from __future__ import annotations

import sys
from typing import Optional


def main(args: Optional[list] = None) -> int:  # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String, Empty
    except ImportError:
        print("[terrain_manager_node] rclpy not available.",
              file=sys.stderr)
        return 1

    rclpy.init(args=args)
    node = Node("terrain_manager_node")
    active = node.declare_parameter(
        "active_terrain", "shackleton_tile").value

    pub = node.create_publisher(String, "/terrain/active", 1)
    msg = String(); msg.data = str(active)

    def on_reset(_msg):
        node.get_logger().info(
            f"reset received; re-announcing terrain='{active}'")
        pub.publish(msg)

    node.create_subscription(Empty, "/dashboard/reset", on_reset, 10)
    node.create_timer(2.0, lambda: pub.publish(msg))
    node.get_logger().info(
        f"terrain_manager_node started, active='{active}'")

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
