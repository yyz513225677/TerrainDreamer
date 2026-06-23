"""lunar_dashboard_bridge — ROS 2 ↔ dashboard WebSocket bridge.

Subscribes to live /lunar_jackal/* topics + tails the Dreamer training
log, and hosts a WebSocket on port 8765 emitting the envelopes the
jackal_dreamer_dashboard frontend already understands:

  {topic: "telemetry", payload: TelemetrySample}
  {topic: "lidar",     payload: LidarSample}
  {topic: "dreamer",   payload: DreamerMetrics}
  {topic: "mode",      payload: {mode, estop}}

Schema is taken from jackal_dreamer_dashboard/dashboard/src/lib/protocol.ts.

Reuse-first: this is a thin glue node. WebSocket via the `websockets`
package (already installed for the original dashboard_bridge), ROS
plumbing via rclpy (system python), no custom protocols.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional


def _quat_to_rpy(x: float, y: float, z: float, w: float
                 ) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 \
        else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


# ───────────────────────────────────────────────────────────────────────────
# Dreamer training log tail — parses per-step lines emitted by
# dreamer_interface_node.
#
#   [INFO] [...] [dreamer_interface_node]: step=1 goal='...' action=(±x m/s, ±y rad/s) goal_obs=['+0.10', ...]
#   [INFO] [...] [dreamer_interface_node]: [train] step 1: wm/total=1.399 ac/actor=1.549 ac/critic=1.419
#   [INFO] [...] [dreamer_interface_node]: [stop-and-turn] ...
# ───────────────────────────────────────────────────────────────────────────

_RE_STEP_HEARTBEAT = re.compile(
    r"step=(\d+).*?action=\(([-+\d.]+) m/s, ([-+\d.]+) rad/s\)"
    r"(?:.*?buf=(\d+)eps trains=(\d+))?",
)
_RE_TRAIN_LOSS = re.compile(
    r"\[train\] step (\d+): wm/total=([-+\d.eE]+) "
    r"ac/actor=([-+\d.eE]+) ac/critic=([-+\d.eE]+)",
)
_RE_SAT = re.compile(r"\[stop-and-turn\] (stopping|turn complete)")


class DreamerLogTail:
    """Background thread that follows the Dreamer log and keeps the
    latest training values in `self.snapshot`."""

    def __init__(self, path: Path):
        self.path = path
        self.snapshot: dict[str, Any] = {
            "action_linear": 0.0,
            "action_angular": 0.0,
            "reward_est": 0.0,
            "value_est": 0.0,
            "uncertainty": 0.0,
            "world_model_loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "bc_loss": 0.0,
            "replay_buffer_size": 0,
            "human_demo_ratio": 0.0,
            "step": 0,
            "train_steps": 0,
            "sat_state": "driving",
        }
        self._t = threading.Thread(target=self._run, daemon=True,
                                   name="dreamer_log_tail")

    def start(self):
        self._t.start()

    def _run(self):
        while True:
            try:
                if not self.path.is_file():
                    time.sleep(1.0)
                    continue
                with self.path.open("r") as fh:
                    fh.seek(0, 2)   # tail -f: start at EOF
                    while True:
                        line = fh.readline()
                        if not line:
                            time.sleep(0.1)
                            continue
                        self._consume(line)
            except Exception:
                time.sleep(1.0)

    def _consume(self, line: str):
        m = _RE_TRAIN_LOSS.search(line)
        if m:
            self.snapshot["train_steps"] = int(m.group(1))
            self.snapshot["world_model_loss"] = float(m.group(2))
            self.snapshot["actor_loss"] = float(m.group(3))
            self.snapshot["critic_loss"] = float(m.group(4))
            return
        m = _RE_STEP_HEARTBEAT.search(line)
        if m:
            self.snapshot["step"] = int(m.group(1))
            self.snapshot["action_linear"] = float(m.group(2))
            self.snapshot["action_angular"] = float(m.group(3))
            if m.group(4):
                self.snapshot["replay_buffer_size"] = int(m.group(4))
            return
        m = _RE_SAT.search(line)
        if m:
            self.snapshot["sat_state"] = (
                "turning" if m.group(1) == "stopping" else "driving")


# ───────────────────────────────────────────────────────────────────────────
# Goals YAML — maps current goal_id → (x, y) for goal_xy field
# ───────────────────────────────────────────────────────────────────────────

def _load_goals_xy(yaml_path: Path) -> dict[str, tuple[float, float]]:
    if not yaml_path.is_file():
        return {}
    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text()) or {}
        return {g["goal_id"]: (float(g["x"]), float(g["y"]))
                for g in data.get("goals", [])}
    except Exception:
        return {}


# ───────────────────────────────────────────────────────────────────────────
# WebSocket hub
# ───────────────────────────────────────────────────────────────────────────

class WSHub:
    def __init__(self):
        self.clients: set[Any] = set()

    async def register(self, ws):
        self.clients.add(ws)

    async def unregister(self, ws):
        self.clients.discard(ws)

    async def broadcast(self, envelope: dict):
        if not self.clients:
            return
        msg = json.dumps(envelope) + "\n"
        dead = []
        for ws in self.clients:
            try:
                await asyncio.wait_for(ws.send(msg), timeout=0.2)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main(args: Optional[list] = None) -> int:  # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Imu, LaserScan
        from nav_msgs.msg import Odometry
        from std_msgs.msg import String, Float32, Bool
        import websockets
    except ImportError as exc:
        print(f"[lunar_dashboard_bridge] missing dep: {exc}",
              file=sys.stderr)
        return 1

    rclpy.init(args=args)
    node = Node("lunar_dashboard_bridge")
    log_path = Path(node.declare_parameter(
        "dreamer_log",
        os.environ.get("DREAMER_LOG", "/tmp/dreamer_train.log")).value)
    ws_port = int(node.declare_parameter("ws_port", 8765).value)
    goals_yaml = Path(node.declare_parameter(
        "goals_yaml",
        "/home/rickslab3/Documents/Leo/terrain_dreamer/"
        "lunar_south_pole_gazebo/config/lunar_goals.yaml").value)
    broadcast_hz = float(node.declare_parameter(
        "broadcast_hz", 10.0).value)

    tail = DreamerLogTail(log_path)
    tail.start()
    goals_xy = _load_goals_xy(goals_yaml)
    node.get_logger().info(
        f"loaded {len(goals_xy)} goals; dreamer log = {log_path}")

    # ROS-side cache
    last = {
        "odom": None, "imu": None, "lidar": None,
        "current_goal": "", "goal_distance": 0.0, "goal_reached": False,
        "collision_count": 0,
    }
    lock = threading.Lock()

    def on_odom(msg):
        with lock:
            last["odom"] = msg

    def on_imu(msg):
        with lock:
            last["imu"] = msg

    def on_scan(msg):
        with lock:
            last["lidar"] = msg

    def on_goal(msg):
        with lock:
            last["current_goal"] = (msg.data or "").strip()

    def on_dist(msg):
        with lock:
            last["goal_distance"] = float(msg.data)

    def on_reach(msg):
        with lock:
            last["goal_reached"] = bool(msg.data)

    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
    # Subscribe to both /lunar_jackal/* (placeholder rover, Phase 2)
    # AND /j100_0001/* (Clearpath J100, current default). The lock
    # serialises writes; the latest message wins.
    node.create_subscription(Odometry, "/lunar_jackal/odom", on_odom, 10)
    node.create_subscription(Imu, "/lunar_jackal/imu", on_imu, qos)
    node.create_subscription(LaserScan, "/lunar_jackal/scan", on_scan, qos)
    node.create_subscription(Odometry, "/j100_0001/platform/odom",
                             on_odom, 10)
    node.create_subscription(Imu, "/j100_0001/sensors/imu_0/data",
                             on_imu, qos)
    node.create_subscription(LaserScan,
                             "/j100_0001/sensors/lidar3d_0/scan",
                             on_scan, qos)
    node.create_subscription(String, "/lunar_jackal/current_goal",
                             on_goal, 10)
    node.create_subscription(Float32, "/lunar_jackal/goal_distance",
                             on_dist, 10)
    node.create_subscription(Bool, "/lunar_jackal/goal_reached",
                             on_reach, 10)

    # Spin rclpy in a thread; asyncio runs the WS server in the main loop.
    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(node), daemon=True,
        name="rclpy_spin_lunar_bridge")
    spin_thread.start()

    hub = WSHub()
    loop = asyncio.new_event_loop()

    async def handler(ws):
        await hub.register(ws)
        node.get_logger().info(
            f"WS client connected ({len(hub.clients)} active)")
        try:
            async for _ in ws:    # ignore inbound (UI doesn't talk back)
                pass
        except Exception:
            pass
        finally:
            await hub.unregister(ws)
            node.get_logger().info(
                f"WS client disconnected ({len(hub.clients)} active)")

    async def producer():
        period = 1.0 / max(broadcast_hz, 1.0)
        while True:
            await asyncio.sleep(period)
            with lock:
                odom = last["odom"]
                imu = last["imu"]
                lidar = last["lidar"]
                cgoal = last["current_goal"]
                gdist = last["goal_distance"]
                greach = last["goal_reached"]

            # ----- telemetry envelope (TelemetrySample) ----------------
            pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
            odom_pl = {"linear_x": 0.0, "angular_z": 0.0}
            if odom is not None:
                p = odom.pose.pose.position
                q = odom.pose.pose.orientation
                _, _, yaw = _quat_to_rpy(q.x, q.y, q.z, q.w)
                pose = {"x": p.x, "y": p.y, "yaw": yaw}
                odom_pl = {
                    "linear_x": odom.twist.twist.linear.x,
                    "angular_z": odom.twist.twist.angular.z,
                }

            imu_pl = {
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
            }
            if imu is not None:
                q = imu.orientation
                r, p, y = _quat_to_rpy(q.x, q.y, q.z, q.w)
                imu_pl = {
                    "roll": r, "pitch": p, "yaw": y,
                    "angular_velocity": {
                        "x": imu.angular_velocity.x,
                        "y": imu.angular_velocity.y,
                        "z": imu.angular_velocity.z,
                    },
                    "linear_acceleration": {
                        "x": imu.linear_acceleration.x,
                        "y": imu.linear_acceleration.y,
                        "z": imu.linear_acceleration.z,
                    },
                }

            goal_xy = goals_xy.get(cgoal)
            telemetry = {
                "timestamp": time.time(),
                "sim_time": time.time(),
                "pose": pose,
                "odom": odom_pl,
                "imu": imu_pl,
                "collision": False,
                "stuck_score": 0.0,
                "rollover_risk": min(1.0, abs(imu_pl["roll"]) / 0.6),
                "collision_count": 0,
                "goal_xy": [goal_xy[0], goal_xy[1]] if goal_xy else None,
                "ros_connected": True,
                "gazebo_connected": True,
                "battery": 1.0,
            }
            await hub.broadcast({"topic": "telemetry",
                                 "payload": telemetry})

            # ----- lidar envelope --------------------------------------
            if lidar is not None and lidar.ranges:
                # Decimate to 360 samples max so the wire payload stays small
                stride = max(1, len(lidar.ranges) // 360)
                ranges = list(lidar.ranges[::stride])
                await hub.broadcast({
                    "topic": "lidar",
                    "payload": {
                        "ranges": ranges,
                        "angle_min": float(lidar.angle_min),
                        "angle_increment": float(lidar.angle_increment) * stride,
                    },
                })

            # ----- dreamer envelope ------------------------------------
            s = tail.snapshot
            await hub.broadcast({
                "topic": "dreamer",
                "payload": {
                    "action_linear":     s["action_linear"],
                    "action_angular":    s["action_angular"],
                    "reward_est":        s["reward_est"],
                    "value_est":         s["value_est"],
                    "uncertainty":       s["uncertainty"],
                    "world_model_loss":  s["world_model_loss"],
                    "actor_loss":        s["actor_loss"],
                    "critic_loss":       s["critic_loss"],
                    "bc_loss":           s["bc_loss"],
                    "replay_buffer_size": s["replay_buffer_size"],
                    "human_demo_ratio":  s["human_demo_ratio"],
                    # extra fields (the dashboard ignores unknown keys)
                    "step":              s["step"],
                    "train_steps":       s["train_steps"],
                    "sat_state":         s["sat_state"],
                },
            })

            # ----- mode envelope (always autonomous, never estopped) ---
            await hub.broadcast({
                "topic": "mode",
                "payload": {
                    "mode": "autonomous",
                    "estop": False,
                },
            })

    async def serve():
        async with websockets.serve(handler, "0.0.0.0", ws_port,
                                    ping_interval=20, ping_timeout=20):
            node.get_logger().info(
                f"WebSocket server listening on ws://0.0.0.0:{ws_port}")
            await producer()

    try:
        loop.run_until_complete(serve())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            loop.close()
        except Exception:
            pass
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
