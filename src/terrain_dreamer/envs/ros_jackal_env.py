"""
RosJackalEnv — Gymnasium wrapper over ROS 2 Jazzy + Gazebo Sim "Harmonic".

Clean rebuild — every fix from the prior bug-stack baked in upfront:

  * Per-instance MultiThreadedExecutor (NOT class-level shared). Adding a
    node to an executor that's already spinning on an empty wait-set
    doesn't reliably wake it; we create the executor + add the node FIRST,
    then start the spin thread.
  * `_teleport()` invalidates the cached `_latest_odom/cloud/imu`. Without
    this, `_wait_until_settled()` returns immediately on a STALE odom
    snapshot from before the teleport, and the first step() sees a bogus
    pose (often near the previous mission's goal) → spurious "reach in
    3 steps over 27 m" detections.
  * `wait_ready()` uses `timeout=3` per topic-echo probe. ros2 topic echo
    needs ~1.2 s of DDS discovery before it can subscribe.
  * Sensor QoS = RELIABLE volatile depth=5 (matches what ros_gz_bridge
    publishes on the GZ→ROS direction).
  * `close()` orders teardown: executor.shutdown → spin.join → remove_node
    → destroy_node → rclpy.shutdown (no "terminate called without an
    active exception" at process exit).
  * Goals are filtered through TraversabilityMask when one is available.

Topics consumed:
    /velodyne_points      sensor_msgs/PointCloud2     bridged from gz
    /imu/data             sensor_msgs/Imu             bridged from gz
    /ground_truth/odom    nav_msgs/Odometry           relayed from /odom

Topic published:
    /cmd_vel              geometry_msgs/Twist         bridged to gz
"""
from __future__ import annotations

import math
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

try:
    import rclpy
    from rclpy.node import Node as RclpyNode
    from rclpy.qos import (
        QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy,
    )
    from rclpy.executors import MultiThreadedExecutor
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu, PointCloud2
    from sensor_msgs_py import point_cloud2 as pc2
except ImportError as e:
    raise ImportError(
        "rclpy / ROS 2 message packages not importable. Source "
        "/opt/ros/jazzy/setup.bash before launching the env. "
        f"Original: {e}"
    )

from terrain_dreamer.envs.sensors.velodyne_vlp32 import PointCloud
from terrain_dreamer.envs.markers import MarkerManager, TraversabilityMask


# ── tunables ────────────────────────────────────────────────────────────────
MAX_LINEAR_VEL   = 0.5         # m/s
MAX_ANGULAR_VEL  = 0.8         # rad/s
LIDAR_MAX_RANGE  = 50.0        # m
MAX_POINTS       = 16384       # padded obs ceiling (32 rings × 512 samples)

GOAL_REACH_DIST  = 0.8
FLIP_PITCH_ROLL  = math.radians(60)
FLIP_GRACE_STEPS = 10
FLIP_POST_TELEPORT_GRACE = 6
MAX_FLIPS_PER_EPISODE = 5
FLIP_SETTLE_TIMEOUT = 1.5
ACTION_LPF_ALPHA  = 0.7
EP_TIMEOUT_STEPS  = 1500


def _yaw_from_q(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _pitch_roll_from_q(q) -> Tuple[float, float]:
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = (math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1
             else math.asin(sinp))
    return pitch, roll


@dataclass
class StepInfo:
    dist_to_goal: float
    flipped: bool
    reached: bool
    timeout: bool
    flip_count: int = 0


_QOS_SENSOR = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
    durability=QoSDurabilityPolicy.VOLATILE,
)


def _gz_set_pose(world: str, model: str,
                  x: float, y: float, z: float, yaw: float,
                  timeout_ms: int = 2000) -> bool:
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    req = (
        f'name: "{model}", '
        f'position: {{ x: {x}, y: {y}, z: {z} }}, '
        f'orientation: {{ x: 0, y: 0, z: {sy}, w: {cy} }}'
    )
    try:
        out = subprocess.run(
            ["gz", "service", "-s", f"/world/{world}/set_pose",
             "--reqtype", "gz.msgs.Pose",
             "--reptype", "gz.msgs.Boolean",
             "--timeout", str(timeout_ms),
             "--req", req],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return "data: true" in out.stdout


class RosJackalEnv(gym.Env):
    """Gazebo Sim + ROS 2 Jackal lunar navigation env. Gymnasium API."""

    metadata = {"render_modes": []}

    # rclpy.init is process-global; share it across instances.
    _rclpy_inited: bool = False
    _init_lock = threading.Lock()

    @classmethod
    def _ensure_rclpy(cls):
        with cls._init_lock:
            if not cls._rclpy_inited:
                rclpy.init(args=None)
                cls._rclpy_inited = True

    def __init__(
        self,
        node_name: str = "ros_jackal_env",
        step_hz: float = 10.0,
        model_name: str = "jackal",
        goal: Optional[Tuple[float, float]] = None,
        max_episode_steps: int = EP_TIMEOUT_STEPS,
        env_name: Optional[str] = None,
        world_name: Optional[str] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.env_name = env_name or "flat"
        # Default world name matches the SDF <world> element.
        # Lunar-terrain envs live under "lunar_south_pole_<terrain>"; flat /
        # varied / mare envs are "moon_<name>".
        if world_name is None:
            if self.env_name in ("default", "rugged", "extreme"):
                world_name = f"lunar_south_pole_{self.env_name}" \
                    if self.env_name != "default" else "lunar_south_pole"
            else:
                world_name = f"moon_{self.env_name}"
        self.world_name = world_name
        self.step_dt = 1.0 / step_hz
        self.max_episode_steps = max_episode_steps
        self._goal = np.array(goal if goal is not None else (5.0, 0.0),
                              dtype=np.float32)

        self._ensure_rclpy()
        self._node = RclpyNode(node_name)

        # Latest-message slots; written from spin thread, read under lock.
        self._lock = threading.Lock()
        self._latest_cloud: Optional[PointCloud] = None
        self._latest_imu:   Optional[Imu]        = None
        self._latest_odom:  Optional[Odometry]   = None

        # Subscriptions — built BEFORE the executor starts spinning so that
        # add_node sees them all and we never spin on an empty wait-set.
        # Topic names match the Clearpath J100 namespace used by the
        # current sim stack (lunar_south_pole_*.launch.py spawns the J100
        # under /j100_0001).
        self._node.create_subscription(
            PointCloud2, "/j100_0001/sensors/lidar3d_0/points",
            self._on_cloud, _QOS_SENSOR)
        self._node.create_subscription(
            Imu, "/j100_0001/sensors/imu_0/data",
            self._on_imu, _QOS_SENSOR)
        self._node.create_subscription(
            Odometry, "/j100_0001/platform/odom",
            self._on_odom, _QOS_SENSOR)

        # cmd_vel published as TwistStamped to match the trainer's
        # publisher (Clearpath's twist_mux subscribes to TwistStamped).
        from geometry_msgs.msg import TwistStamped as _TS
        self._cmd_pub = self._node.create_publisher(
            _TS, "/j100_0001/cmd_vel", 1)

        # Per-instance executor + spin thread.
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name=f"rclpy_spin_{node_name}",
            daemon=True,
        )
        self._spin_thread.start()

        try:
            self._markers: Optional[MarkerManager] = MarkerManager(
                env_name=self.env_name, world_name=self.world_name,
            )
        except Exception as e:
            self._node.get_logger().warn(f"[env] markers disabled: {e}")
            self._markers = None

        self._traversable: Optional[TraversabilityMask] = (
            TraversabilityMask.find_for(self.env_name)
        )
        if self._traversable is not None:
            print(f"[env] traversability mask loaded ({self.env_name}, "
                  f"{100.0 * self._traversable.mask.mean():.1f}% drivable)")
        else:
            print(f"[env] no traversability mask for env={self.env_name!r}; "
                  "goals will not be drivability-filtered")

        self.observation_space = spaces.Dict({
            "points":   spaces.Box(-LIDAR_MAX_RANGE, LIDAR_MAX_RANGE,
                                    shape=(MAX_POINTS, 4), dtype=np.float32),
            "n_points": spaces.Box(0, MAX_POINTS, shape=(), dtype=np.int32),
            "imu":      spaces.Box(-np.inf, np.inf,
                                    shape=(6,), dtype=np.float32),
            "pose":     spaces.Box(-np.inf, np.inf,
                                    shape=(3,), dtype=np.float32),
            "goal_obs": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._step_count = 0
        self._flip_count = 0
        self._last_flip_step = -10**9
        self._prev_dist_to_goal: Optional[float] = None
        self._cmd_lpf = np.zeros(2, dtype=np.float32)

    # ─── subscribers ────────────────────────────────────────────────────
    def _on_cloud(self, msg: PointCloud2):
        try:
            structured = pc2.read_points(
                msg, field_names=("x", "y", "z", "intensity"),
                skip_nans=True,
            )
            arr = np.asarray(structured)
            if arr.size == 0:
                pts = np.zeros((0, 4), dtype=np.float32)
            else:
                pts = np.stack(
                    [arr["x"], arr["y"], arr["z"], arr["intensity"]],
                    axis=-1,
                ).astype(np.float32)
        except Exception:
            pts = np.zeros((0, 4), dtype=np.float32)
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cloud = PointCloud(timestamp=ts, points=pts)
        with self._lock:
            self._latest_cloud = cloud

    def _on_imu(self, msg: Imu):
        with self._lock:
            self._latest_imu = msg

    def _on_odom(self, msg: Odometry):
        with self._lock:
            self._latest_odom = msg

    # ─── helpers ────────────────────────────────────────────────────────
    def _wait_until_settled(self, timeout: float = 8.0,
                              vel_thresh: float = 0.15):
        """Block until a NEW odom message arrives with low velocity. Since
        _teleport invalidates _latest_odom, this reliably waits for a
        fresh post-teleport pose, not a stale snapshot."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                odom = self._latest_odom
            if odom is not None:
                lv, av = odom.twist.twist.linear, odom.twist.twist.angular
                speed = math.sqrt(lv.x**2 + lv.y**2 + lv.z**2)
                omega = math.sqrt(av.x**2 + av.y**2 + av.z**2)
                if speed < vel_thresh and omega < vel_thresh:
                    return
            time.sleep(0.05)

    def _current_tilt(self) -> float:
        with self._lock:
            odom = self._latest_odom
        if odom is None:
            return 0.0
        pitch, roll = _pitch_roll_from_q(odom.pose.pose.orientation)
        return max(abs(pitch), abs(roll))

    def _wait_for_fresh_data(self, timeout: float = 5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                ok = (self._latest_cloud is not None
                      and self._latest_imu is not None
                      and self._latest_odom is not None)
            if ok:
                return
            time.sleep(0.02)
        raise RuntimeError(
            "Timed out waiting for /velodyne_points + /imu/data + "
            "/ground_truth/odom. Is `ros2 launch terrain_dreamer_bringup "
            "moon_jackal.launch.py` running?"
        )

    def _find_level_spawn(self, x: float, y: float, yaw: float,
                            *, tilt_ok: float = math.radians(20),
                            max_retries: int = 8,
                            nudge_m: float = 1.5) -> Tuple[float, float, float]:
        attempt_x, attempt_y = x, y
        for k in range(max_retries + 1):
            self._teleport(attempt_x, attempt_y, yaw)
            self._wait_until_settled(timeout=8.0, vel_thresh=0.15)
            tilt = self._current_tilt()
            if tilt < tilt_ok:
                if k > 0:
                    print(f"[spawn] level after {k} retries: "
                          f"({attempt_x:+.1f},{attempt_y:+.1f}) "
                          f"tilt={math.degrees(tilt):.1f}°")
                return (attempt_x, attempt_y, yaw)
            radius = nudge_m * (1 + 0.5 * k)
            angle  = k * 2.39996
            attempt_x = x + radius * math.cos(angle)
            attempt_y = y + radius * math.sin(angle)
        print(f"[spawn] WARN: no level spot near ({x:+.1f},{y:+.1f}) "
              f"after {max_retries} tries — last tilt={math.degrees(tilt):.1f}°")
        return (attempt_x, attempt_y, yaw)

    def _make_obs(self) -> Dict[str, np.ndarray]:
        with self._lock:
            cloud, imu, odom = self._latest_cloud, self._latest_imu, self._latest_odom

        pts = cloud.points if cloud is not None else np.zeros((0, 4),
                                                                dtype=np.float32)
        n = min(pts.shape[0], MAX_POINTS)
        padded = np.zeros((MAX_POINTS, 4), dtype=np.float32)
        if n > 0:
            padded[:n] = pts[:n]

        if imu is not None:
            w, a = imu.angular_velocity, imu.linear_acceleration
            imu_vec = np.array([w.x, w.y, w.z, a.x, a.y, a.z],
                                dtype=np.float32)
        else:
            imu_vec = np.zeros(6, dtype=np.float32)

        if odom is not None:
            p, q = odom.pose.pose.position, odom.pose.pose.orientation
            pose = np.array([p.x, p.y, _yaw_from_q(q)], dtype=np.float32)
        else:
            pose = np.zeros(3, dtype=np.float32)

        dx = self._goal[0] - pose[0]
        dy = self._goal[1] - pose[1]
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        heading_err = math.atan2(math.sin(bearing - pose[2]),
                                  math.cos(bearing - pose[2]))
        goal_obs = np.array([
            np.clip(dx / 30.0, -1.0, 1.0),
            np.clip(dy / 30.0, -1.0, 1.0),
            np.clip(dist / 30.0,  0.0, 1.0),
            np.clip(heading_err / math.pi, -1.0, 1.0),
        ], dtype=np.float32)

        return {"points": padded, "n_points": np.int32(n),
                "imu": imu_vec, "pose": pose, "goal_obs": goal_obs}

    def _teleport(self, x: float, y: float, yaw: float, z: float = 12.0):
        """Teleport via gz transport AND invalidate stale sensor cache."""
        self._publish_stop_stamped()
        with self._lock:
            self._latest_odom  = None
            self._latest_cloud = None
            self._latest_imu   = None
        _gz_set_pose(self.world_name, self.model_name, x, y, z, yaw)

    # ─── Gymnasium API ──────────────────────────────────────────────────
    def set_goal(self, goal: Tuple[float, float]):
        self._goal = np.array(goal, dtype=np.float32)

    def sample_drivable_goal(
        self, rng, max_dist: float,
        origin: Optional[Tuple[float, float]] = None,
        min_dist: float = 0.0,
    ) -> Tuple[float, float]:
        if origin is None:
            with self._lock:
                odom = self._latest_odom
            if odom is not None:
                p = odom.pose.pose.position
                origin = (float(p.x), float(p.y))
            else:
                origin = (0.0, 0.0)
        if self._traversable is not None:
            g = self._traversable.sample_drivable_goal(
                rng, max_dist=max_dist, origin=origin, min_dist=min_dist)
            if g is not None:
                return g
        r = float(rng.uniform(min_dist, max_dist))
        theta = float(rng.uniform(-math.pi, math.pi))
        return (origin[0] + r * math.cos(theta),
                origin[1] + r * math.sin(theta))

    def reset(
        self, *, seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        spawn_x   = float(options.get("spawn_x", 0.0))
        spawn_y   = float(options.get("spawn_y", 0.0))
        spawn_yaw = float(options.get("spawn_yaw", 0.0))
        goal = options.get("goal", None)
        if goal is not None:
            self._goal = np.array(goal, dtype=np.float32)

        self._publish_stop_stamped()
        self._wait_for_fresh_data(timeout=5.0)
        ax, ay, ayaw = self._find_level_spawn(spawn_x, spawn_y, spawn_yaw)
        self._step_count = 0
        self._flip_count = 0
        self._last_flip_step = -10**9
        self._prev_dist_to_goal = None
        self._cmd_lpf[:] = 0.0

        if self._markers is not None:
            try:
                self._markers.update_start_goal(
                    start_xy=(ax, ay),
                    goal_xy=(float(self._goal[0]), float(self._goal[1])),
                )
            except Exception as e:
                self._node.get_logger().warn(f"[env] marker update failed: {e}")

        obs = self._make_obs()
        info = {
            "goal":           self._goal.copy(),
            "spawn_xy":       np.array([ax, ay], dtype=np.float32),
            "spawn_yaw":      float(ayaw),
            "spawn_tilt_deg": math.degrees(self._current_tilt()),
        }
        return obs, info

    def _publish_stop_stamped(self):
        from geometry_msgs.msg import TwistStamped as _TS
        m = _TS()
        m.header.stamp = self._node.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        self._cmd_pub.publish(m)

    def step(self, action: np.ndarray):
        from geometry_msgs.msg import TwistStamped as _TS
        target = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._cmd_lpf = (ACTION_LPF_ALPHA * self._cmd_lpf
                         + (1.0 - ACTION_LPF_ALPHA) * target).astype(np.float32)
        cmd = _TS()
        cmd.header.stamp = self._node.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.twist.linear.x  = float(self._cmd_lpf[0]) * MAX_LINEAR_VEL
        cmd.twist.angular.z = float(self._cmd_lpf[1]) * MAX_ANGULAR_VEL
        self._cmd_pub.publish(cmd)
        time.sleep(self.step_dt)
        self._step_count += 1

        obs = self._make_obs()
        pose = obs["pose"]
        dx, dy = self._goal[0] - pose[0], self._goal[1] - pose[1]
        dist = math.hypot(dx, dy)

        # Reward: 4× distance-reduction shaping − time penalty − tilt cost
        # +30 on reach, −15 on flip (in-range for reward predictor).
        if self._prev_dist_to_goal is None:
            shaping = 0.0
        else:
            shaping = (self._prev_dist_to_goal - dist) * 4.0
        self._prev_dist_to_goal = dist
        time_penalty = 0.03

        with self._lock:
            odom = self._latest_odom
        flipped = False
        tilt_cost = 0.0
        in_grace = (self._step_count <= FLIP_GRACE_STEPS
                    or self._step_count - self._last_flip_step
                    <= FLIP_POST_TELEPORT_GRACE)
        if odom is not None and not in_grace:
            pitch, roll = _pitch_roll_from_q(odom.pose.pose.orientation)
            flipped = (abs(pitch) > FLIP_PITCH_ROLL
                       or abs(roll) > FLIP_PITCH_ROLL)
            tilt_deg = max(abs(math.degrees(pitch)), abs(math.degrees(roll)))
            if tilt_deg > 30.0:
                t = min((tilt_deg - 30.0) / 30.0, 1.0)
                tilt_cost = 0.5 * t * t

        reached  = dist < GOAL_REACH_DIST
        timeout  = self._step_count >= self.max_episode_steps
        flip_limit_hit = False

        reward = shaping - time_penalty - tilt_cost
        if reached:
            reward += 30.0
        if flipped:
            reward -= 15.0
            self._flip_count += 1
            self._last_flip_step = self._step_count
            flip_limit_hit = self._flip_count >= MAX_FLIPS_PER_EPISODE
            if not flip_limit_hit:
                self._publish_stop_stamped()
                self._teleport(float(pose[0]), float(pose[1]),
                                float(pose[2]))
                self._wait_until_settled(
                    timeout=FLIP_SETTLE_TIMEOUT, vel_thresh=0.2)
                self._cmd_lpf[:] = 0.0
                with self._lock:
                    odom2 = self._latest_odom
                if odom2 is not None:
                    p = odom2.pose.pose.position
                    self._prev_dist_to_goal = math.hypot(
                        self._goal[0] - p.x, self._goal[1] - p.y)

        terminated = reached
        truncated  = (timeout or flip_limit_hit) and not terminated
        info = StepInfo(dist_to_goal=dist, flipped=flipped,
                         reached=reached, timeout=timeout,
                         flip_count=self._flip_count).__dict__
        info["flip_limit_hit"] = bool(flip_limit_hit)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        try:
            self._publish_stop_stamped()
        except Exception:
            pass
        # Ordered teardown — stop the executor before destroying the node so
        # we don't get "terminate called without an active exception" at
        # process exit.
        try: self._executor.shutdown()
        except Exception: pass
        try: self._spin_thread.join(timeout=2.0)
        except Exception: pass
        try: self._executor.remove_node(self._node)
        except Exception: pass
        try: self._node.destroy_node()
        except Exception: pass

    def wait_ready(self, timeout: float = 30.0):
        self._wait_for_fresh_data(timeout=timeout)
