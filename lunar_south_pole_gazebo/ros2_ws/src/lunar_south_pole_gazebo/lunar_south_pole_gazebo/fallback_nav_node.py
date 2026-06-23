"""fallback_nav_node — pure-pursuit + reactive lidar obstacle avoidance.

Publishes /j100_0001/cmd_vel (TwistStamped — Clearpath uses stamped Twist)
based on:
  * /j100_0001/platform/odom         — current pose
  * /mission/current_goal            — target pose
  * /j100_0001/sensors/lidar3d_0/scan — 2-D laser slice for reactive avoid
                                       (the gpu_lidar publishes scan + points;
                                       the scan is one ring's worth which is
                                       perfect for a flat-ground reactive
                                       controller).

This is the publish-or-fallback controller: it runs unconditionally,
producing a *safe* heading command toward the goal while braking and
veering away from forward obstacles. Dreamer can override by publishing
on /j100_0001/cmd_vel_dreamer (priority via twist_mux), or by simply
republishing on /j100_0001/cmd_vel at a higher rate.

Not a deep-learning policy — a 50-line classical controller meant to
keep the mission demo working while Dreamer fine-tunes in parallel.
"""
from __future__ import annotations

import math
import sys
from typing import Optional


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def pure_pursuit_cmd(
    rx: float, ry: float, yaw: float,
    gx: float, gy: float,
    *, max_lin: float = 0.4, max_ang: float = 0.7,
    angular_gain: float = 1.8, slowdown_radius: float = 2.0,
) -> tuple[float, float]:
    """Pure-pursuit (lin, ang) toward goal."""
    dx, dy = gx - rx, gy - ry
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return 0.0, 0.0
    bearing = math.atan2(dy, dx)
    heading_err = math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))
    # angular command proportional to heading error
    ang = max(-max_ang, min(max_ang, angular_gain * heading_err))
    # linear command: slow down when far off heading or close to goal
    forward_alignment = max(0.0, math.cos(heading_err))   # 1 when facing goal
    proximity_scale = min(1.0, dist / slowdown_radius)
    lin = max_lin * forward_alignment * proximity_scale
    return lin, ang


def reactive_avoid(
    lin: float, ang: float, ranges, angle_min: float, angle_inc: float,
    *, brake_dist: float = 1.5, stop_dist: float = 0.6,
    forward_arc_rad: float = 1.0,
) -> tuple[float, float]:
    """Brake/veer if obstacles are in the forward cone."""
    if ranges is None or len(ranges) == 0:
        return lin, ang
    min_left = float("inf")
    min_right = float("inf")
    min_front = float("inf")
    for i, r in enumerate(ranges):
        if not (0.05 < r < 80.0):
            continue
        a = angle_min + i * angle_inc
        if abs(a) > forward_arc_rad:
            continue
        if r < min_front:
            min_front = r
        if a > 0 and r < min_left:
            min_left = r
        if a < 0 and r < min_right:
            min_right = r
    if min_front == float("inf"):
        return lin, ang
    # Hard stop if too close
    if min_front < stop_dist:
        return 0.0, ang
    # Linearly brake between brake_dist and stop_dist
    if min_front < brake_dist:
        brake = (min_front - stop_dist) / (brake_dist - stop_dist)
        lin *= max(0.0, brake)
        # Veer away from the closer side
        if min_left < min_right:
            ang -= 0.5   # turn right (negative angular)
        else:
            ang += 0.5
    return lin, ang


def main(args: Optional[list] = None) -> int:  # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
        from sensor_msgs.msg import LaserScan
    except ImportError as exc:
        print(f"[fallback_nav_node] missing dep: {exc}", file=sys.stderr)
        return 1

    rclpy.init(args=args)
    node = Node("fallback_nav_node")

    odom_topic = node.declare_parameter(
        "odom_topic", "/lunar_jackal/odom").value
    scan_topic = node.declare_parameter(
        "scan_topic", "/lunar_jackal/scan").value
    cmd_topic = node.declare_parameter(
        "cmd_topic", "/lunar_jackal/cmd_vel").value
    cmd_stamped = bool(node.declare_parameter("cmd_stamped", False).value)
    max_lin = float(node.declare_parameter("max_linear_mps", 0.4).value)
    max_ang = float(node.declare_parameter("max_angular_radps", 0.7).value)
    rate_hz = float(node.declare_parameter("rate_hz", 10.0).value)

    state = {"pose_xy": None, "yaw": 0.0, "goal": None, "scan": None}

    qos_sensor = QoSProfile(depth=10,
                            reliability=ReliabilityPolicy.BEST_EFFORT)

    def on_odom(msg):
        state["pose_xy"] = (msg.pose.pose.position.x,
                            msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        state["yaw"] = _yaw_from_quat(q.x, q.y, q.z, q.w)

    def on_goal(msg):
        state["goal"] = (msg.pose.position.x, msg.pose.position.y)

    def on_scan(msg):
        state["scan"] = msg

    node.create_subscription(Odometry, odom_topic, on_odom, 10)
    node.create_subscription(PoseStamped, "/mission/current_goal",
                             on_goal, 10)
    node.create_subscription(LaserScan, scan_topic, on_scan, qos_sensor)

    msg_type = TwistStamped if cmd_stamped else Twist
    pub = node.create_publisher(msg_type, cmd_topic, 10)

    def make_cmd(lin: float, ang: float):
        if cmd_stamped:
            ts = TwistStamped()
            ts.header.stamp = node.get_clock().now().to_msg()
            ts.twist.linear.x = float(lin)
            ts.twist.angular.z = float(ang)
            return ts
        t = Twist()
        t.linear.x = float(lin)
        t.angular.z = float(ang)
        return t

    def tick():
        if state["pose_xy"] is None or state["goal"] is None:
            return
        rx, ry = state["pose_xy"]
        gx, gy = state["goal"]
        lin, ang = pure_pursuit_cmd(
            rx, ry, state["yaw"], gx, gy,
            max_lin=max_lin, max_ang=max_ang,
        )
        s = state["scan"]
        if s is not None:
            lin, ang = reactive_avoid(
                lin, ang, list(s.ranges),
                float(s.angle_min), float(s.angle_increment))
        pub.publish(make_cmd(lin, ang))

    node.create_timer(1.0 / max(rate_hz, 1.0), tick)
    node.get_logger().info(
        f"fallback_nav_node — publishing {cmd_topic} (max_lin={max_lin}, "
        f"max_ang={max_ang})")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        pub.publish(make_cmd(0.0, 0.0))  # zero twist on shutdown
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
