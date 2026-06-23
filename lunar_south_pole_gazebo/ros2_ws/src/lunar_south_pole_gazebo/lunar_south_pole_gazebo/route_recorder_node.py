"""route_recorder_node — synchronise /odom + /imu, augment with the
latest /cmd_vel and goal state, write one JSONL sample per sync event
into data/routes/.

The route is recorded from *odometry*, NOT from the IMU — the IMU
provides orientation, angular velocity and linear acceleration that
are merged into the same sample.

Reuse: `message_filters.ApproximateTimeSynchronizer` is the standard
ROS 2 sync primitive. Output uses stdlib json + PyYAML. No custom
bag format.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Pure function — z-axis rotation from a quaternion. Used by tests."""
    import math
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def make_sample(*, timestamp: float, route_id: str, robot_name: str,
                pose: dict, odom: dict, imu: dict, cmd_vel: dict,
                goal: dict, collision: bool) -> dict:
    """Pure-function sample builder — used by tests to validate schema."""
    return {
        "timestamp": float(timestamp),
        "route_id": str(route_id),
        "robot_name": str(robot_name),
        "pose": {k: float(pose.get(k, 0.0))
                 for k in ("x", "y", "z", "yaw")},
        "odom": {k: float(odom.get(k, 0.0))
                 for k in ("linear_x", "linear_y", "angular_z")},
        "imu": {
            "orientation": [float(v) for v in imu.get(
                "orientation", [0.0, 0.0, 0.0, 1.0])],
            "angular_velocity": [float(v) for v in imu.get(
                "angular_velocity", [0.0, 0.0, 0.0])],
            "linear_acceleration": [float(v) for v in imu.get(
                "linear_acceleration", [0.0, 0.0, 0.0])],
        },
        "cmd_vel": {k: float(cmd_vel.get(k, 0.0))
                    for k in ("linear_x", "angular_z")},
        "goal": {
            "goal_id": str(goal.get("goal_id", "")),
            "distance_to_goal": float(goal.get("distance_to_goal", 0.0)),
            "goal_reached": bool(goal.get("goal_reached", False)),
        },
        "collision": bool(collision),
    }


def main(args: list | None = None) -> int:  # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        import message_filters
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu
        from geometry_msgs.msg import Twist
        from std_msgs.msg import String, Float32, Bool
        import yaml
    except ImportError:
        print("[route_recorder_node] rclpy/message_filters not available",
              file=sys.stderr)
        return 1

    rclpy.init(args=args)
    node = Node("route_recorder_node")

    robot_name = node.declare_parameter("robot_name", "lunar_jackal").value
    odom_topic = node.declare_parameter(
        "odom_topic", "/lunar_jackal/odom").value
    imu_topic = node.declare_parameter(
        "imu_topic", "/lunar_jackal/imu").value
    cmd_vel_topic = node.declare_parameter(
        "cmd_vel_topic", "/lunar_jackal/cmd_vel").value
    routes_dir = Path(node.declare_parameter(
        "routes_dir",
        os.environ.get("LUNAR_ROUTES_DIR",
                       str(Path.cwd() / "data" / "routes"))).value)
    route_id_param = node.declare_parameter("route_id", "auto").value
    slop_s = float(node.declare_parameter("slop_s", 0.05).value)
    queue_size = int(node.declare_parameter("queue_size", 20).value)
    # Clearpath J100 uses TwistStamped on /cmd_vel; original placeholder
    # rover uses bare Twist. Switch via this param.
    cmd_vel_stamped = bool(node.declare_parameter(
        "cmd_vel_stamped", False).value)

    route_id = (route_id_param if route_id_param and route_id_param != "auto"
                else "route_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    routes_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = routes_dir / f"{route_id}.jsonl"
    meta_path = routes_dir / f"{route_id}.yaml"
    jsonl_fh = jsonl_path.open("w", buffering=1)  # line-buffered

    # Latched, non-stamped, or low-rate inputs — cached.
    cache = {
        "cmd_vel": {"linear_x": 0.0, "angular_z": 0.0},
        "goal": {"goal_id": "", "distance_to_goal": 0.0,
                 "goal_reached": False},
        "collision": False,
        "samples_written": 0,
        "started_at": _dt.datetime.now().isoformat(),
    }

    # Whole-dict assignment (not nested-key mutation) so the sync
    # callback's snapshot is always self-consistent even under a
    # MultiThreadedExecutor.
    def on_cmd(msg):
        cache["cmd_vel"] = {
            "linear_x": float(msg.linear.x),
            "angular_z": float(msg.angular.z),
        }

    def on_current(msg):
        cache["goal"] = {**cache["goal"], "goal_id": msg.data or ""}

    def on_dist(msg):
        cache["goal"] = {**cache["goal"],
                         "distance_to_goal": float(msg.data)}

    def on_reach(msg):
        cache["goal"] = {**cache["goal"],
                         "goal_reached": bool(msg.data)}

    def on_collision(msg):
        cache["collision"] = bool(msg.data)

    if cmd_vel_stamped:
        from geometry_msgs.msg import TwistStamped
        def on_cmd_stamped(msg):
            on_cmd(msg.twist)
        node.create_subscription(TwistStamped, cmd_vel_topic,
                                 on_cmd_stamped, 10)
    else:
        node.create_subscription(Twist, cmd_vel_topic, on_cmd, 10)
    node.create_subscription(String, "/lunar_jackal/current_goal",
                             on_current, 10)
    node.create_subscription(Float32, "/lunar_jackal/goal_distance",
                             on_dist, 10)
    node.create_subscription(Bool, "/lunar_jackal/goal_reached",
                             on_reach, 10)
    node.create_subscription(Bool, "/collision_status", on_collision, 10)

    # message_filters Subscribers + ApproximateTimeSynchronizer for the
    # two stamped, high-rate inputs.
    sub_odom = message_filters.Subscriber(node, Odometry, odom_topic)
    sub_imu = message_filters.Subscriber(node, Imu, imu_topic)
    sync = message_filters.ApproximateTimeSynchronizer(
        [sub_odom, sub_imu], queue_size=queue_size, slop=slop_s)

    def on_pair(odom_msg, imu_msg):
        q = odom_msg.pose.pose.orientation
        sample = make_sample(
            timestamp=odom_msg.header.stamp.sec
                      + odom_msg.header.stamp.nanosec * 1e-9,
            route_id=route_id,
            robot_name=robot_name,
            pose={
                "x": odom_msg.pose.pose.position.x,
                "y": odom_msg.pose.pose.position.y,
                "z": odom_msg.pose.pose.position.z,
                "yaw": _quat_to_yaw(q.x, q.y, q.z, q.w),
            },
            odom={
                "linear_x": odom_msg.twist.twist.linear.x,
                "linear_y": odom_msg.twist.twist.linear.y,
                "angular_z": odom_msg.twist.twist.angular.z,
            },
            imu={
                "orientation": [imu_msg.orientation.x,
                                imu_msg.orientation.y,
                                imu_msg.orientation.z,
                                imu_msg.orientation.w],
                "angular_velocity": [imu_msg.angular_velocity.x,
                                     imu_msg.angular_velocity.y,
                                     imu_msg.angular_velocity.z],
                "linear_acceleration": [imu_msg.linear_acceleration.x,
                                        imu_msg.linear_acceleration.y,
                                        imu_msg.linear_acceleration.z],
            },
            cmd_vel=dict(cache["cmd_vel"]),
            goal=dict(cache["goal"]),
            collision=bool(cache["collision"]),
        )
        jsonl_fh.write(json.dumps(sample) + "\n")
        cache["samples_written"] += 1

    sync.registerCallback(on_pair)
    node.get_logger().info(
        f"route_recorder_node writing {jsonl_path} "
        f"(slop={slop_s}s queue={queue_size})")

    written = {"meta": False}

    def write_meta():
        if written["meta"]:
            return  # idempotent — survives both signal + KeyboardInterrupt
        meta = {
            "route_id": route_id,
            "robot_name": robot_name,
            "jsonl": str(jsonl_path),
            "started_at": cache["started_at"],
            "ended_at": _dt.datetime.now().isoformat(),
            "samples_written": cache["samples_written"],
            "topics": {
                "odom": odom_topic,
                "imu": imu_topic,
                "cmd_vel": cmd_vel_topic,
                "current_goal": "/lunar_jackal/current_goal",
                "goal_distance": "/lunar_jackal/goal_distance",
                "goal_reached": "/lunar_jackal/goal_reached",
                "collision_status": "/collision_status",
            },
        }
        meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
        try:
            jsonl_fh.flush()
            jsonl_fh.close()
        except Exception:
            pass
        written["meta"] = True

    # Register so SIGTERM / SIGINT both flush metadata before rclpy is
    # torn down. atexit covers normal Python exit.
    import atexit
    import signal
    atexit.register(write_meta)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, lambda *_: (write_meta(),
                                           rclpy.shutdown()))
        except Exception:
            pass

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        write_meta()
        try:
            node.get_logger().info(
                f"route_recorder_node wrote {cache['samples_written']} "
                f"samples → {jsonl_path}; meta → {meta_path}")
        except Exception:
            pass
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
