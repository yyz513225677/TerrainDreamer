"""dashboard_bridge_node — websocket <-> ROS 2 multiplexer.

Listens on ws://0.0.0.0:8765 (configurable via --port). Subscribes to a
handful of ROS topics, throttles + reshapes them into newline-delimited
JSON envelopes, and forwards each envelope to every connected client.
Demultiplexes incoming client commands into ROS publications.

Supports a --mock flag which bypasses ROS entirely and feeds the WS
server from a pure-Python synthetic generator (see mock_generator.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import threading
import time
from typing import Any, Optional

try:
    import websockets  # type: ignore
except ImportError as e:  # pragma: no cover
    sys.stderr.write(
        "[dashboard_bridge] python 'websockets' package missing. "
        "Install with: pip install websockets\n")
    raise

from .mock_generator import telemetry_stream


# --- protocol constants ------------------------------------------------
WS_HOST_DEFAULT = "0.0.0.0"
WS_PORT_DEFAULT = 8765
TELEMETRY_MAX_HZ = 20.0
LIDAR_TARGET_RAYS = 180


# =====================================================================
# WS server core — works identically in mock and live mode
# =====================================================================
class WSHub:
    """Tracks connected clients and broadcasts envelopes to all of them."""

    def __init__(self) -> None:
        self._clients: set[Any] = set()
        self._lock = asyncio.Lock()

    async def register(self, ws: Any) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def unregister(self, ws: Any) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, envelope: dict) -> None:
        if not self._clients:
            return
        line = json.dumps(envelope) + "\n"
        # Snapshot under lock, send outside lock.
        async with self._lock:
            dead: list[Any] = []
            targets = list(self._clients)
        for ws in targets:
            try:
                await ws.send(line)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    def client_count(self) -> int:
        return len(self._clients)


# =====================================================================
# Mock runner — no ROS at all
# =====================================================================
async def _mock_publisher(hub: WSHub, command_q: "asyncio.Queue[dict]") -> None:
    """Push synthetic envelopes to hub at ~20 Hz; consume commands from q."""
    gen = telemetry_stream(rate_hz=TELEMETRY_MAX_HZ)
    period = 1.0 / TELEMETRY_MAX_HZ
    next_tick = time.monotonic()

    # Drain commands without blocking publishing.
    async def _drain_commands() -> None:
        while True:
            cmd = await command_q.get()
            # Mock mode: echo mode commands back so the UI sees state flip.
            if cmd.get("cmd") == "set_mode":
                await hub.broadcast({
                    "topic": "mode",
                    "payload": {"mode": cmd.get("mode", "idle"),
                                "estop": False},
                })
            elif cmd.get("cmd") == "estop":
                await hub.broadcast({
                    "topic": "mode",
                    "payload": {"mode": "estop",
                                "estop": bool(cmd.get("active", True))},
                })
            elif cmd.get("cmd") == "reset":
                await hub.broadcast({
                    "topic": "mode",
                    "payload": {"mode": "idle", "estop": False},
                })

    drain_task = asyncio.create_task(_drain_commands())
    try:
        while True:
            try:
                envelope = next(gen)
            except StopIteration:
                break
            await hub.broadcast(envelope)
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                # We fell behind; resync.
                next_tick = time.monotonic()
    finally:
        drain_task.cancel()


# =====================================================================
# Live ROS runner
# =====================================================================
class RosBridge:
    """Imports rclpy lazily so --mock doesn't require ROS at all."""

    def __init__(self, hub: WSHub,
                 command_q: "asyncio.Queue[dict]",
                 loop: asyncio.AbstractEventLoop,
                 scan_topic: str = "/scan") -> None:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Bool, Empty, String
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import LaserScan, Imu
        from nav_msgs.msg import Odometry
        from jackal_dreamer_dashboard_msgs.msg import (
            DashboardMode, DreamerMetrics, CollisionStatus,
        )

        self._rclpy = rclpy
        self._hub = hub
        self._loop = loop
        self._command_q = command_q

        rclpy.init()
        self._node = Node("dashboard_bridge_node")
        n = self._node

        # State accumulator for telemetry packets.
        self._latest_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self._latest_odom = {"linear_x": 0.0, "angular_z": 0.0}
        self._latest_imu = {"rpy": [0.0, 0.0, 0.0],
                            "angvel": [0.0, 0.0, 0.0],
                            "linacc": [0.0, 0.0, 0.0]}
        self._latest_collision = {"in_contact": False, "stuck": 0.0}
        self._last_telemetry_publish = 0.0
        self._telemetry_period = 1.0 / TELEMETRY_MAX_HZ

        # --- subscribers -------------------------------------------------
        n.create_subscription(LaserScan, scan_topic, self._on_scan, 10)
        n.create_subscription(Imu, "/imu/data", self._on_imu, 10)
        n.create_subscription(Odometry, "/ground_truth/odom",
                              self._on_odom, 10)
        n.create_subscription(DreamerMetrics, "/dreamer/metrics",
                              self._on_dreamer_metrics, 10)
        n.create_subscription(Twist, "/dreamer/action",
                              self._on_dreamer_action, 10)
        n.create_subscription(CollisionStatus, "/collision_status",
                              self._on_collision, 10)
        n.create_subscription(DashboardMode, "/dashboard/mode/current",
                              self._on_mode, 10)

        # --- publishers --------------------------------------------------
        self._pub_mode = n.create_publisher(
            DashboardMode, "/dashboard/mode", 10)
        self._pub_estop = n.create_publisher(
            Bool, "/dashboard/estop", 10)
        self._pub_reset = n.create_publisher(
            Empty, "/dashboard/reset", 10)
        self._pub_manual = n.create_publisher(
            Twist, "/dashboard/manual_cmd", 10)
        self._pub_rec_start = n.create_publisher(
            String, "/dashboard/recording/start", 10)
        self._pub_rec_stop = n.create_publisher(
            Empty, "/dashboard/recording/stop", 10)
        self._pub_rec_save = n.create_publisher(
            String, "/dashboard/recording/save", 10)
        self._pub_replay_start = n.create_publisher(
            String, "/dashboard/replay/start", 10)
        self._pub_replay_stop = n.create_publisher(
            Empty, "/dashboard/replay/stop", 10)
        self._pub_train_start = n.create_publisher(
            Empty, "/dashboard/training/start", 10)

        # Keep references for outbound publishing in the command loop.
        self._DashboardMode = DashboardMode
        self._Bool = Bool
        self._Empty = Empty
        self._String = String
        self._Twist = Twist

        # Background command consumer.
        self._cmd_task = asyncio.run_coroutine_threadsafe(
            self._command_loop(), self._loop)

        # rclpy spin thread.
        self._spin_thread = threading.Thread(
            target=self._spin, daemon=True, name="rclpy-spin")
        self._spin_thread.start()

    # ------------- rclpy spin -----------------------------------------
    def _spin(self) -> None:
        try:
            self._rclpy.spin(self._node)
        except Exception:
            pass

    def shutdown(self) -> None:
        try:
            self._node.destroy_node()
        finally:
            try:
                self._rclpy.try_shutdown()
            except Exception:
                pass

    # ------------- subscriber callbacks --------------------------------
    def _on_scan(self, msg) -> None:
        ranges = list(msg.ranges)
        # Decimate to LIDAR_TARGET_RAYS.
        n = len(ranges)
        if n > LIDAR_TARGET_RAYS:
            step = n / LIDAR_TARGET_RAYS
            ranges = [ranges[int(i * step)] for i in range(LIDAR_TARGET_RAYS)]
        # Replace inf/nan with angle_max (or 0) so JSON serializes cleanly.
        clean: list[float] = []
        for r in ranges:
            if r is None or math.isnan(r) or math.isinf(r):
                clean.append(0.0)
            else:
                clean.append(float(r))
        payload = {
            "ranges": clean,
            "angle_min": float(msg.angle_min),
            "angle_max": float(msg.angle_max),
        }
        self._enqueue({"topic": "lidar", "payload": payload})

    def _on_imu(self, msg) -> None:
        q = msg.orientation
        # Convert quaternion -> rpy.
        roll, pitch, yaw = _quat_to_rpy(q.x, q.y, q.z, q.w)
        self._latest_imu = {
            "rpy": [roll, pitch, yaw],
            "angvel": [msg.angular_velocity.x,
                       msg.angular_velocity.y,
                       msg.angular_velocity.z],
            "linacc": [msg.linear_acceleration.x,
                       msg.linear_acceleration.y,
                       msg.linear_acceleration.z],
        }
        self._maybe_emit_telemetry()

    def _on_odom(self, msg) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = _quat_to_rpy(q.x, q.y, q.z, q.w)
        self._latest_pose = {"x": float(p.x), "y": float(p.y),
                             "yaw": float(yaw)}
        self._latest_odom = {
            "linear_x": float(msg.twist.twist.linear.x),
            "angular_z": float(msg.twist.twist.angular.z),
        }
        self._maybe_emit_telemetry()

    def _on_collision(self, msg) -> None:
        self._latest_collision = {
            "in_contact": bool(msg.in_contact),
            "stuck": float(msg.stuck_score),
        }
        self._maybe_emit_telemetry()

    def _on_dreamer_metrics(self, msg) -> None:
        payload = {
            "wm_loss": float(msg.wm_loss),
            "actor_loss": float(msg.actor_loss),
            "critic_loss": float(msg.critic_loss),
            "bc_loss": float(msg.bc_loss),
            "buffer_size": int(msg.buffer_size),
            "human_ratio": float(msg.human_ratio),
            "reward_est": float(msg.reward_est),
            "value_est": float(msg.value_est),
            "uncertainty_est": float(msg.uncertainty_est),
            "action_linear": float(msg.action_linear),
            "action_angular": float(msg.action_angular),
        }
        self._enqueue({"topic": "dreamer", "payload": payload})

    def _on_dreamer_action(self, msg) -> None:
        # Reported as part of telemetry stream; just record nothing here.
        # If desired we could publish a dedicated topic, but `dreamer`
        # already carries action_linear/action_angular.
        pass

    def _on_mode(self, msg) -> None:
        self._enqueue({
            "topic": "mode",
            "payload": {"mode": msg.mode, "estop": bool(msg.estop_active)},
        })

    # ------------- outbound throttle ----------------------------------
    def _maybe_emit_telemetry(self) -> None:
        now = time.monotonic()
        if now - self._last_telemetry_publish < self._telemetry_period:
            return
        self._last_telemetry_publish = now
        self._enqueue({
            "topic": "telemetry",
            "payload": {
                "stamp": now,
                "pose": self._latest_pose,
                "odom": self._latest_odom,
                "imu_rpy": self._latest_imu["rpy"],
                "imu_angvel": self._latest_imu["angvel"],
                "imu_linacc": self._latest_imu["linacc"],
                "collision": self._latest_collision["in_contact"],
                "stuck": self._latest_collision["stuck"],
            },
        })

    def _enqueue(self, envelope: dict) -> None:
        # Cross-thread: rclpy callbacks run in the spin thread; the WS
        # broadcaster runs on the asyncio loop in the main thread.
        asyncio.run_coroutine_threadsafe(
            self._hub.broadcast(envelope), self._loop)

    # ------------- inbound command demuxer ---------------------------
    async def _command_loop(self) -> None:
        while True:
            cmd = await self._command_q.get()
            try:
                self._dispatch_command(cmd)
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().warn(
                    f"Failed to dispatch command {cmd}: {exc}")

    def _dispatch_command(self, cmd: dict) -> None:
        op = cmd.get("cmd")
        if op == "set_mode":
            msg = self._DashboardMode()
            msg.mode = str(cmd.get("mode", "idle"))
            msg.estop_active = False
            msg.seq = 0
            self._pub_mode.publish(msg)
        elif op == "estop":
            self._pub_estop.publish(self._Bool(data=bool(cmd.get("active", True))))
        elif op == "reset":
            self._pub_reset.publish(self._Empty())
        elif op == "manual_cmd":
            tw = self._Twist()
            tw.linear.x = float(cmd.get("linear", 0.0))
            tw.angular.z = float(cmd.get("angular", 0.0))
            self._pub_manual.publish(tw)
        elif op == "recording":
            action = cmd.get("action")
            route_id = str(cmd.get("route_id", ""))
            if action == "start":
                payload = json.dumps({
                    "route_id": route_id,
                    "operator_id": str(cmd.get("operator_id", "")),
                    "environment_id": str(cmd.get("environment_id", "")),
                })
                self._pub_rec_start.publish(self._String(data=payload))
            elif action == "stop":
                self._pub_rec_stop.publish(self._Empty())
            elif action == "save":
                self._pub_rec_save.publish(self._String(data=route_id))
        elif op == "replay":
            action = cmd.get("action")
            route_id = str(cmd.get("route_id", ""))
            if action == "start":
                self._pub_replay_start.publish(self._String(data=route_id))
            elif action == "stop":
                self._pub_replay_stop.publish(self._Empty())
        elif op == "training":
            self._pub_train_start.publish(self._Empty())


# =====================================================================
# helpers
# =====================================================================
def _quat_to_rpy(x: float, y: float, z: float, w: float
                 ) -> tuple[float, float, float]:
    # ZYX intrinsic.
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1 \
        else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


# =====================================================================
# main
# =====================================================================
async def _ws_handler(ws: Any, hub: WSHub,
                      command_q: "asyncio.Queue[dict]") -> None:
    await hub.register(ws)
    try:
        async for raw in ws:
            try:
                cmd = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(cmd, dict) and "cmd" in cmd:
                await command_q.put(cmd)
    except Exception:
        pass
    finally:
        await hub.unregister(ws)


async def _run_async(host: str, port: int, mock: bool,
                     scan_topic: str) -> None:
    hub = WSHub()
    command_q: "asyncio.Queue[dict]" = asyncio.Queue()
    loop = asyncio.get_running_loop()

    bridge: Optional[RosBridge] = None
    if not mock:
        bridge = RosBridge(hub, command_q, loop, scan_topic=scan_topic)
        print(f"[dashboard_bridge] live mode, scan_topic={scan_topic}",
              flush=True)
    else:
        asyncio.create_task(_mock_publisher(hub, command_q))
        print("[dashboard_bridge] MOCK mode — no ROS subscribers",
              flush=True)

    async def handler(ws: Any) -> None:
        await _ws_handler(ws, hub, command_q)

    print(f"[dashboard_bridge] listening on ws://{host}:{port}",
          flush=True)
    async with websockets.serve(handler, host, port):
        try:
            await asyncio.Future()  # run forever
        finally:
            if bridge is not None:
                bridge.shutdown()


def main(argv: Optional[list[str]] = None) -> None:
    # rclpy may eat --ros-args; strip those before parsing.
    args_in = list(sys.argv[1:] if argv is None else argv)
    if "--ros-args" in args_in:
        idx = args_in.index("--ros-args")
        args_in = args_in[:idx]

    p = argparse.ArgumentParser(description="Jackal-Dreamer dashboard bridge.")
    p.add_argument("--mock", action="store_true",
                   help="Bypass ROS and emit synthetic telemetry.")
    p.add_argument("--host", default=WS_HOST_DEFAULT)
    p.add_argument("--port", type=int, default=WS_PORT_DEFAULT)
    p.add_argument("--scan-topic", default="/scan",
                   help="LiDAR topic to decimate (e.g. /scan or "
                        "/velodyne_points).")
    args = p.parse_args(args_in)

    try:
        asyncio.run(_run_async(args.host, args.port, args.mock,
                               args.scan_topic))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
