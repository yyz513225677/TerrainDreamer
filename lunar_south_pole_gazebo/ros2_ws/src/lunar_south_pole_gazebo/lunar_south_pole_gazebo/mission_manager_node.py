"""mission_manager_node — random outbound goal → flag → reach → return
home with flag → repeat.

FSM:
  IDLE → GO_OUT → REACHED_OUT → GO_HOME → REACHED_HOME → IDLE → (new goal)
  └─────── spawns yellow flag ──────┘└── spawns red home flag ─┘

Outputs (consumed by Dreamer / fallback controller / route recorder /
dashboard):
  /mission/current_goal    (geometry_msgs/PoseStamped — target xy)
  /mission/state            (std_msgs/String)
  /mission/route_id         (std_msgs/String — incremented each loop)
  /lunar_jackal/current_goal (std_msgs/String — legacy compat with
                              dreamer node + route recorder)
  /lunar_jackal/goal_distance (std_msgs/Float32)
  /lunar_jackal/goal_reached  (std_msgs/Bool)

The flag entity is spawned in gz via subprocess call to
`gz service -s /world/<world>/create` (no extra ROS dependency).
Removal uses `/world/<world>/remove`.
"""
from __future__ import annotations

import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# ───────────────────────────────────────────────────────────────────────────
# Flag SDF template — coloured per-spawn via {color_rgba}
# ───────────────────────────────────────────────────────────────────────────
FLAG_SDF_TEMPLATE = """<?xml version="1.0"?>
<sdf version="1.10">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="pole">
        <pose>0 0 1.0 0 0 0</pose>
        <geometry><cylinder><radius>0.04</radius><length>2.0</length></cylinder></geometry>
        <material>
          <ambient>0.95 0.95 0.95 1</ambient>
          <diffuse>0.95 0.95 0.95 1</diffuse>
        </material>
      </visual>
      <visual name="cloth">
        <pose>0.30 0 1.65 0 0 0</pose>
        <geometry><box><size>0.60 0.02 0.40</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
          <emissive>{er} {eg} {eb} 1</emissive>
        </material>
      </visual>
      <visual name="base">
        <pose>0 0 0.01 0 0 0</pose>
        <geometry><cylinder><radius>0.25</radius><length>0.02</length></cylinder></geometry>
        <material><ambient>0.20 0.20 0.20 1</ambient></material>
      </visual>
    </link>
  </model>
</sdf>
"""


def make_flag_sdf(name: str, rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    return FLAG_SDF_TEMPLATE.format(
        name=name, r=r, g=g, b=b,
        er=r * 0.25, eg=g * 0.25, eb=b * 0.25,
    )


def gz_spawn_flag(world: str, name: str, x: float, y: float,
                  rgb: tuple[float, float, float]) -> bool:
    """Spawn a flag via gz service /world/<world>/create. Returns True on success."""
    # protobuf text-format sdf: '...' must not contain newlines — compact whitespace
    sdf = " ".join(make_flag_sdf(name, rgb).split())
    # DEM surface sits around z=3-4 m in this world; spawn the flag base at
    # z=4 so the pole and cloth are visible above the terrain. The flag is
    # static, so it stays where placed.
    req = (
        f'sdf: \'{sdf}\' '
        f'name: "{name}" '
        f'allow_renaming: true '
        f'pose {{ position {{ x: {x} y: {y} z: 4.0 }} '
        f'orientation {{ w: 1 }} }}'
    )
    try:
        r = subprocess.run(
            ["gz", "service", "-s", f"/world/{world}/create",
             "--reqtype", "gz.msgs.EntityFactory",
             "--reptype", "gz.msgs.Boolean",
             "--timeout", "2000",
             "--req", req],
            capture_output=True, text=True, timeout=5)
        return "data: true" in r.stdout
    except Exception:
        return False


def gz_remove_entity(world: str, name: str) -> bool:
    """Remove a model from the world by name."""
    req = f'name: "{name}" type: MODEL'
    try:
        r = subprocess.run(
            ["gz", "service", "-s", f"/world/{world}/remove",
             "--reqtype", "gz.msgs.Entity",
             "--reptype", "gz.msgs.Boolean",
             "--timeout", "2000",
             "--req", req],
            capture_output=True, text=True, timeout=5)
        return "data: true" in r.stdout
    except Exception:
        return False


# ───────────────────────────────────────────────────────────────────────────
# Random goal sampler
# ───────────────────────────────────────────────────────────────────────────
def sample_random_goal(min_radius: float, max_radius: float,
                       rng: random.Random) -> tuple[float, float]:
    """Uniform in an annulus around the origin."""
    angle = rng.uniform(-math.pi, math.pi)
    r = math.sqrt(rng.uniform(min_radius * min_radius,
                              max_radius * max_radius))
    return (r * math.cos(angle), r * math.sin(angle))


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────
def main(args: Optional[list] = None) -> int:  # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, DurabilityPolicy
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import String, Float32, Bool
    except ImportError as exc:
        print(f"[mission_manager_node] missing dep: {exc}", file=sys.stderr)
        return 1

    rclpy.init(args=args)
    node = Node("mission_manager_node")

    world = node.declare_parameter("world", "lunar_south_pole").value
    odom_topic = node.declare_parameter(
        "odom_topic", "/j100_0001/platform/odom").value
    reach_radius = float(node.declare_parameter("reach_radius_m", 1.5).value)
    min_radius = float(node.declare_parameter("min_radius_m", 5.0).value)
    max_radius = float(node.declare_parameter("max_radius_m", 15.0).value)
    home_xy = (
        float(node.declare_parameter("home_x", 0.0).value),
        float(node.declare_parameter("home_y", 0.0).value),
    )
    settle_time = float(node.declare_parameter("settle_time_s", 3.0).value)
    seed = int(node.declare_parameter("seed", 0).value)
    rng = random.Random(seed if seed != 0 else None)

    latched = QoSProfile(depth=1)
    latched.durability = DurabilityPolicy.TRANSIENT_LOCAL

    pub_goal_pose = node.create_publisher(
        PoseStamped, "/mission/current_goal", latched)
    pub_state = node.create_publisher(String, "/mission/state", latched)
    pub_route_id = node.create_publisher(String, "/mission/route_id", latched)
    # Legacy compat — the existing dreamer + route_recorder + dashboard
    # already subscribe to these.
    pub_goal_id = node.create_publisher(
        String, "/lunar_jackal/current_goal", latched)
    pub_goal_dist = node.create_publisher(
        Float32, "/lunar_jackal/goal_distance", 10)
    pub_goal_reached = node.create_publisher(
        Bool, "/lunar_jackal/goal_reached", 10)

    state = {
        "pose": None,           # (x, y) from odom
        "fsm": "IDLE",          # IDLE | GO_OUT | REACHED_OUT | GO_HOME | REACHED_HOME
        "current_goal": None,   # (x, y)
        "loop_id": 0,
        "flag_name": None,      # name of currently-spawned flag in gz
        "reached_at": None,     # wall-clock time of reach event
    }

    def on_odom(msg: Odometry):
        state["pose"] = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    node.create_subscription(Odometry, odom_topic, on_odom, 10)

    def publish_status(stamp):
        if state["current_goal"] is None or state["pose"] is None:
            return
        gx, gy = state["current_goal"]
        px, py = state["pose"]
        d = math.hypot(gx - px, gy - py)
        pub_goal_dist.publish(Float32(data=float(d)))
        pub_goal_reached.publish(Bool(data=bool(d <= reach_radius)))

    def remove_current_flag():
        if state["flag_name"]:
            gz_remove_entity(world, state["flag_name"])
            node.get_logger().info(f"[mission] removed flag {state['flag_name']}")
        state["flag_name"] = None

    def enter_state(new_state: str):
        old = state["fsm"]
        state["fsm"] = new_state
        msg = String(); msg.data = new_state
        pub_state.publish(msg)
        node.get_logger().info(f"[mission] {old} → {new_state}")

    def set_current_goal(xy: tuple[float, float], goal_id: str,
                         flag_rgb: tuple[float, float, float], flag_name: str):
        state["current_goal"] = xy
        ps = PoseStamped()
        ps.header.frame_id = "world"
        ps.pose.position.x = float(xy[0])
        ps.pose.position.y = float(xy[1])
        ps.pose.orientation.w = 1.0
        pub_goal_pose.publish(ps)
        pub_goal_id.publish(String(data=goal_id))
        # Spawn flag in gz
        remove_current_flag()
        ok = gz_spawn_flag(world, flag_name, xy[0], xy[1], flag_rgb)
        if ok:
            state["flag_name"] = flag_name
            node.get_logger().info(
                f"[mission] spawned {flag_name} flag @ ({xy[0]:.2f}, {xy[1]:.2f})")
        else:
            node.get_logger().warn(f"[mission] failed to spawn {flag_name}")

    def begin_outbound():
        state["loop_id"] += 1
        rid = f"loop_{state['loop_id']:03d}"
        pub_route_id.publish(String(data=rid))
        goal_xy = sample_random_goal(min_radius, max_radius, rng)
        set_current_goal(goal_xy, f"goal_out_{state['loop_id']}",
                         (0.96, 0.78, 0.10),   # yellow
                         f"goal_flag_out_{state['loop_id']}")
        enter_state("GO_OUT")

    def begin_return():
        set_current_goal(home_xy, f"goal_home_{state['loop_id']}",
                         (0.92, 0.20, 0.20),   # red
                         f"goal_flag_home_{state['loop_id']}")
        enter_state("GO_HOME")

    def tick():
        now = node.get_clock().now().to_msg()
        # Always publish status so downstream nodes can drive.
        publish_status(now)

        if state["pose"] is None:
            return

        fsm = state["fsm"]
        if fsm == "IDLE":
            begin_outbound()
            return

        # check distance to current goal
        gx, gy = state["current_goal"]
        px, py = state["pose"]
        d = math.hypot(gx - px, gy - py)

        if fsm == "GO_OUT" and d <= reach_radius:
            enter_state("REACHED_OUT")
            state["reached_at"] = time.time()
            return
        if fsm == "REACHED_OUT":
            if time.time() - state["reached_at"] >= settle_time:
                begin_return()
            return
        if fsm == "GO_HOME" and d <= reach_radius:
            enter_state("REACHED_HOME")
            state["reached_at"] = time.time()
            return
        if fsm == "REACHED_HOME":
            if time.time() - state["reached_at"] >= settle_time:
                remove_current_flag()
                enter_state("IDLE")
            return

    node.create_timer(0.2, tick)
    node.get_logger().info(
        f"mission_manager_node started; world={world} "
        f"radius=[{min_radius}, {max_radius}] reach={reach_radius}m")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        remove_current_flag()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
