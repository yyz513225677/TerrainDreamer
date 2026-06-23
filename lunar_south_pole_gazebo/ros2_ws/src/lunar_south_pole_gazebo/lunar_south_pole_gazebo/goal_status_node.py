"""goal_status_node — computes distance to the active goal from /odom
and publishes current_goal / goal_distance / goal_reached.

Goal advancement is *optional* and disabled by default. If the
parameter `advance_on_reach` is true, the node walks to the next goal
in YAML order whenever the rover arrives. Otherwise the active goal
must be changed via the `/lunar_jackal/set_goal` topic
(std_msgs/String — payload = goal_id).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

# Reuse loader from goal_marker_node — single source of truth for
# the goal YAML schema validation.
from lunar_south_pole_gazebo.goal_marker_node import (load_goals,
                                                     _resolve_goals_yaml_default)


def euclidean_distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Pure function — used by tests."""
    return math.hypot(ax - bx, ay - by)


def main(args: list | None = None) -> int:  # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, QoSDurabilityPolicy
        from nav_msgs.msg import Odometry
        from std_msgs.msg import String, Float32, Bool
    except ImportError:
        print("[goal_status_node] rclpy not available", file=sys.stderr)
        return 1

    rclpy.init(args=args)
    node = Node("goal_status_node")
    goals_yaml = node.declare_parameter(
        "goals_yaml", _resolve_goals_yaml_default()).value
    advance_on_reach = bool(node.declare_parameter(
        "advance_on_reach", False).value)

    try:
        goals, defaults = load_goals(Path(goals_yaml))
    except (FileNotFoundError, ValueError) as exc:
        node.get_logger().error(str(exc))
        goals, defaults = [], {}

    import yaml
    try:
        raw = yaml.safe_load(Path(goals_yaml).read_text()) or {}
        initial_id = raw.get("active_goal_id")
    except Exception:
        initial_id = None

    default_arrival_radius_m = float(defaults.get("arrival_radius_m", 3.0))
    by_id = {g["goal_id"]: g for g in goals}
    order = [g["goal_id"] for g in goals]
    state = {"active": initial_id if initial_id in by_id else
             (order[0] if order else None)}

    latched = QoSProfile(depth=1)
    latched.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

    pub_cur = node.create_publisher(String, "/lunar_jackal/current_goal", latched)
    pub_dist = node.create_publisher(Float32, "/lunar_jackal/goal_distance", 10)
    pub_reach = node.create_publisher(Bool, "/lunar_jackal/goal_reached", 10)

    last_xy = {"x": 0.0, "y": 0.0, "have": False}

    def on_odom(msg):
        last_xy["x"] = msg.pose.pose.position.x
        last_xy["y"] = msg.pose.pose.position.y
        last_xy["have"] = True

    def on_set_goal(msg):
        gid = (msg.data or "").strip()
        if gid in by_id:
            state["active"] = gid
            node.get_logger().info(f"active goal → {gid}")
        else:
            node.get_logger().warn(f"unknown goal_id: {gid!r}")

    node.create_subscription(Odometry, "/lunar_jackal/odom", on_odom, 10)
    node.create_subscription(String, "/lunar_jackal/set_goal", on_set_goal, 10)

    if state["active"] is not None:
        pub_cur.publish(String(data=state["active"]))

    def tick():
        if state["active"] is None or not last_xy["have"]:
            return
        g = by_id[state["active"]]
        d = euclidean_distance(last_xy["x"], last_xy["y"],
                               float(g["x"]), float(g["y"]))
        pub_dist.publish(Float32(data=float(d)))
        radius = float(g.get("arrival_radius_m", default_arrival_radius_m))
        reached = d <= radius
        pub_reach.publish(Bool(data=bool(reached)))
        if reached and advance_on_reach:
            try:
                idx = order.index(state["active"])
                if idx + 1 < len(order):
                    state["active"] = order[idx + 1]
                    pub_cur.publish(String(data=state["active"]))
                    # Republish goal_reached=False against the NEW goal
                    # immediately so subscribers don't latch onto a
                    # stale True from the previous tick.
                    pub_reach.publish(Bool(data=False))
                    node.get_logger().info(
                        f"reached; advancing to {state['active']}")
            except ValueError:
                pass

    node.create_timer(0.1, tick)  # 10 Hz status update
    node.get_logger().info(
        f"goal_status_node started; goals={len(goals)} "
        f"active={state['active']!r} advance_on_reach={advance_on_reach}")

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
