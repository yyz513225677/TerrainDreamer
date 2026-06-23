"""hazard_status_node — aggregates collision + tilt signals into one
boolean topic /hazard/active.

Subscribes:
  /collision_status   (std_msgs/Bool)        from a contact sensor, if present
  /imu                (sensor_msgs/Imu)      from the bridge

Publishes:
  /hazard/active      (std_msgs/Bool)        true if collision OR roll>tilt_limit
"""
from __future__ import annotations

import math
import sys
from typing import Optional


def _quat_to_roll(x: float, y: float, z: float, w: float) -> float:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    return math.atan2(sinr_cosp, cosr_cosp)


def main(args: Optional[list] = None) -> int:  # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Imu
        from std_msgs.msg import Bool
    except ImportError:
        print("[hazard_status_node] rclpy not available.",
              file=sys.stderr)
        return 1

    rclpy.init(args=args)
    node = Node("hazard_status_node")
    tilt_limit_rad = float(node.declare_parameter(
        "tilt_limit_rad", math.radians(35.0)).value)

    state = {"collision": False, "roll": 0.0}
    pub = node.create_publisher(Bool, "/hazard/active", 1)

    def on_collision(msg):
        state["collision"] = bool(msg.data)

    def on_imu(msg):
        q = msg.orientation
        state["roll"] = _quat_to_roll(q.x, q.y, q.z, q.w)

    node.create_subscription(Bool, "/collision_status", on_collision, 10)
    node.create_subscription(Imu, "/imu", on_imu, 10)

    def tick():
        hazard = state["collision"] or abs(state["roll"]) > tilt_limit_rad
        pub.publish(Bool(data=bool(hazard)))

    node.create_timer(0.1, tick)
    node.get_logger().info(
        f"hazard_status_node started; tilt_limit={tilt_limit_rad:.3f} rad")

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
