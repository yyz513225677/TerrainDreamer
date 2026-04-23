"""
RosJackalEnv — Gymnasium wrapper over ROS 1 Noetic + Gazebo Classic 11.

Replaces the old PyBullet-based LunarJackalEnv. Assumes Gazebo is already up
via `roslaunch terrain_dreamer_bringup moon_jackal.launch`. This env only
talks ROS — it does NOT launch Gazebo itself. The training launcher
(`run_auto.sh`) handles process orchestration.

Topics consumed:
    /velodyne_points      sensor_msgs/PointCloud2    32-ring LiDAR
    /imu/data             sensor_msgs/Imu            IMU (VN-100 stand-in)
    /ground_truth/odom    nav_msgs/Odometry          Gazebo p3d pose

Topics published:
    /cmd_vel              geometry_msgs/Twist        (lin_vel, ang_vel)

Services used:
    /gazebo/set_model_state     reset the rover
    /gazebo/pause_physics       pause between resets (optional)
    /gazebo/unpause_physics
    /gazebo/reset_world         reset world time (optional)
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

# ROS imports — fail loudly if the user hasn't sourced /opt/ros/noetic/setup.bash.
try:
    import rospy
    from geometry_msgs.msg import Twist, Pose, Point, Quaternion
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu, PointCloud2
    from sensor_msgs import point_cloud2 as pc2
    from gazebo_msgs.msg import ModelState
    from gazebo_msgs.srv import SetModelState, SetModelStateRequest
    from std_srvs.srv import Empty
except ImportError as e:
    raise ImportError(
        "rospy/ROS message packages not importable. Source /opt/ros/noetic/setup.bash "
        f"before launching the env. Original error: {e}"
    )

from terrain_dreamer.envs.sensors.velodyne_vlp32 import PointCloud


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_LINEAR_VEL  = 1.0   # m/s   (Jackal capped)
MAX_ANGULAR_VEL = 2.0   # rad/s
LIDAR_MAX_RANGE = 50.0  # m — matches the xacro
MAX_POINTS      = 16384  # obs padding ceiling (32 rings × 512 samples)

GOAL_REACH_DIST = 0.8   # m
FLIP_PITCH_ROLL = math.radians(60)  # ±60° = flipped
EP_TIMEOUT_STEPS = 600  # matches old PyBullet budget


def _yaw_from_quaternion(q) -> float:
    """ROS quaternion → yaw (Z-axis rotation)."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _pitch_roll_from_quaternion(q) -> Tuple[float, float]:
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    return pitch, roll


@dataclass
class StepInfo:
    dist_to_goal: float
    flipped: bool
    reached: bool
    timeout: bool


class RosJackalEnv(gym.Env):
    """Gazebo+ROS Jackal lunar navigation env. Gymnasium API.

    Assumes an external `roslaunch terrain_dreamer_bringup moon_jackal.launch`
    is already running. Call `reset()` to teleport the rover to spawn and set a
    new goal; `step(action)` publishes a Twist and returns the next obs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        node_name: str = "ros_jackal_env",
        step_hz: float = 10.0,
        model_name: str = "jackal",
        goal: Optional[Tuple[float, float]] = None,
        max_episode_steps: int = EP_TIMEOUT_STEPS,
    ):
        super().__init__()

        self.model_name = model_name
        self.step_dt = 1.0 / step_hz
        self.max_episode_steps = max_episode_steps
        self._goal = np.array(goal if goal is not None else (5.0, 0.0), dtype=np.float32)

        # ROS init — safe to call once per process. anonymous=False so it's
        # easy to see in `rosnode list`; disable_signals=True so Ctrl-C at the
        # Python level still works.
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=False, disable_signals=True)

        # Subscribers — each writes into a latest-message slot under a lock.
        self._lock = threading.Lock()
        self._latest_cloud: Optional[PointCloud] = None
        self._latest_imu: Optional[Imu] = None
        self._latest_odom: Optional[Odometry] = None

        self._cloud_sub = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self._on_cloud, queue_size=1,
            buff_size=2**22,
        )
        self._imu_sub = rospy.Subscriber(
            "/imu/data", Imu, self._on_imu, queue_size=5,
        )
        self._odom_sub = rospy.Subscriber(
            "/ground_truth/odom", Odometry, self._on_odom, queue_size=5,
        )

        # Publisher + services.
        self._cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.wait_for_service("/gazebo/set_model_state", timeout=30.0)
        self._set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self._pause   = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self._unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)

        # Spaces. Dict obs — matches the Dreamer encoder's expected inputs.
        self.observation_space = spaces.Dict({
            "points":    spaces.Box(-LIDAR_MAX_RANGE, LIDAR_MAX_RANGE,
                                     shape=(MAX_POINTS, 4), dtype=np.float32),
            "n_points":  spaces.Box(0, MAX_POINTS, shape=(), dtype=np.int32),
            "imu":       spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "pose":      spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "goal_obs":  spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._step_count = 0
        self._prev_dist_to_goal: Optional[float] = None

    # ---- ROS callbacks ----------------------------------------------------

    def _on_cloud(self, msg: PointCloud2):
        pts = np.array(list(pc2.read_points(
            msg, field_names=("x", "y", "z", "intensity"), skip_nans=True,
        )), dtype=np.float32)
        if pts.size == 0:
            pts = np.zeros((0, 4), dtype=np.float32)
        elif pts.shape[1] == 3:  # some drivers omit intensity
            pts = np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)], axis=1)
        cloud = PointCloud(timestamp=msg.header.stamp.to_sec(), points=pts)
        with self._lock:
            self._latest_cloud = cloud

    def _on_imu(self, msg: Imu):
        with self._lock:
            self._latest_imu = msg

    def _on_odom(self, msg: Odometry):
        with self._lock:
            self._latest_odom = msg

    # ---- Helpers ----------------------------------------------------------

    def _wait_until_settled(self, timeout: float = 8.0, vel_thresh: float = 0.15):
        """Block until the rover has essentially stopped moving after a
        teleport. Prevents bogus flip detections on the first step of an
        episode when the chassis is still bouncing from a drop-in."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                odom = self._latest_odom
            if odom is not None:
                lv = odom.twist.twist.linear
                av = odom.twist.twist.angular
                speed  = math.sqrt(lv.x * lv.x + lv.y * lv.y + lv.z * lv.z)
                omega  = math.sqrt(av.x * av.x + av.y * av.y + av.z * av.z)
                if speed < vel_thresh and omega < vel_thresh:
                    return
            time.sleep(0.05)
        # Not an error — just log and continue; physics may never fully rest.

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
            "Timed out waiting for /velodyne_points, /imu/data, /ground_truth/odom. "
            "Is roslaunch terrain_dreamer_bringup moon_jackal.launch running?"
        )

    def _make_obs(self) -> Dict[str, np.ndarray]:
        with self._lock:
            cloud = self._latest_cloud
            imu = self._latest_imu
            odom = self._latest_odom

        # Points — pad/truncate to MAX_POINTS.
        pts = cloud.points if cloud is not None else np.zeros((0, 4), dtype=np.float32)
        n = min(pts.shape[0], MAX_POINTS)
        padded = np.zeros((MAX_POINTS, 4), dtype=np.float32)
        if n > 0:
            padded[:n] = pts[:n]

        # IMU — angular velocity (3) + linear acceleration (3).
        if imu is not None:
            w = imu.angular_velocity
            a = imu.linear_acceleration
            imu_vec = np.array([w.x, w.y, w.z, a.x, a.y, a.z], dtype=np.float32)
        else:
            imu_vec = np.zeros(6, dtype=np.float32)

        # Pose — (x, y, yaw) in world frame from ground-truth odometry.
        if odom is not None:
            p = odom.pose.pose.position
            q = odom.pose.pose.orientation
            yaw = _yaw_from_quaternion(q)
            pose = np.array([p.x, p.y, yaw], dtype=np.float32)
        else:
            pose = np.zeros(3, dtype=np.float32)

        # Goal observation — matches DreamerActor/Buffer contract (goal_dim=4):
        #   (dx_norm, dy_norm, dist_norm, heading_err_norm)
        # Distances normalized by 30 m (typical goal span), heading by pi.
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

        return {
            "points":    padded,
            "n_points":  np.int32(n),
            "imu":       imu_vec,
            "pose":      pose,
            "goal_obs":  goal_obs,
        }

    def _pose_matrix(self, pos: Tuple[float, float, float],
                     yaw: float) -> Pose:
        p = Pose()
        p.position = Point(x=pos[0], y=pos[1], z=pos[2])
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        p.orientation = Quaternion(x=0.0, y=0.0, z=sy, w=cy)
        return p

    def _teleport(self, x: float, y: float, yaw: float, z: float = 12.0):
        """Teleport the rover. Z defaults well above the heightmap so the rover
        drops cleanly onto the surface (heightmap Z scale is 10 m)."""
        req = SetModelStateRequest()
        ms = ModelState()
        ms.model_name = self.model_name
        ms.pose = self._pose_matrix((x, y, z), yaw)
        # Zero velocities so the previous mission's motion doesn't leak through.
        ms.twist.linear.x = ms.twist.linear.y = ms.twist.linear.z = 0.0
        ms.twist.angular.x = ms.twist.angular.y = ms.twist.angular.z = 0.0
        ms.reference_frame = "world"
        req.model_state = ms
        self._set_state(req)

    # ---- Gymnasium API ----------------------------------------------------

    def set_goal(self, goal: Tuple[float, float]):
        """Change the navigation goal without resetting the rover."""
        self._goal = np.array(goal, dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        spawn_x = float(options.get("spawn_x", 0.0))
        spawn_y = float(options.get("spawn_y", 0.0))
        spawn_yaw = float(options.get("spawn_yaw", 0.0))
        goal = options.get("goal", None)
        if goal is not None:
            self._goal = np.array(goal, dtype=np.float32)

        # Zero-cmd the rover, teleport well above the terrain, let it drop
        # and settle. 12 m free-fall under -1.62 m/s² takes up to ~3.8 s, then
        # we need time for the suspension to damp out. Poll the odom and wait
        # until the rover is effectively stationary.
        self._cmd_pub.publish(Twist())
        self._teleport(spawn_x, spawn_y, spawn_yaw)
        self._wait_for_fresh_data(timeout=5.0)
        self._wait_until_settled(timeout=8.0, vel_thresh=0.15)
        self._step_count = 0
        self._prev_dist_to_goal = None

        obs = self._make_obs()
        info = {"goal": self._goal.copy()}
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        cmd = Twist()
        cmd.linear.x = float(action[0]) * MAX_LINEAR_VEL
        cmd.angular.z = float(action[1]) * MAX_ANGULAR_VEL
        self._cmd_pub.publish(cmd)

        # Let the sim run one step_dt. ROS takes care of message arrival; we
        # just sleep wall-time since Gazebo is running at real-time 1.0.
        time.sleep(self.step_dt)
        self._step_count += 1

        obs = self._make_obs()
        pose = obs["pose"]
        dx = self._goal[0] - pose[0]
        dy = self._goal[1] - pose[1]
        dist = math.hypot(dx, dy)

        # Reward = shaping (distance reduction) + terminal bonuses.
        if self._prev_dist_to_goal is None:
            shaping = 0.0
        else:
            shaping = (self._prev_dist_to_goal - dist) * 2.0
        self._prev_dist_to_goal = dist

        # Flip detection from IMU orientation (via ground truth odom quaternion).
        with self._lock:
            odom = self._latest_odom
        flipped = False
        if odom is not None:
            pitch, roll = _pitch_roll_from_quaternion(odom.pose.pose.orientation)
            flipped = abs(pitch) > FLIP_PITCH_ROLL or abs(roll) > FLIP_PITCH_ROLL

        reached = dist < GOAL_REACH_DIST
        timeout = self._step_count >= self.max_episode_steps

        reward = shaping
        if reached:
            reward += 25.0
        if flipped:
            reward -= 50.0

        terminated = reached or flipped
        truncated = timeout and not terminated

        info = StepInfo(
            dist_to_goal=dist, flipped=flipped,
            reached=reached, timeout=timeout,
        ).__dict__

        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        try:
            self._cmd_pub.publish(Twist())
        except Exception:
            pass

    # ---- convenience for the training loop -------------------------------

    def wait_ready(self, timeout: float = 30.0):
        """Block until Gazebo publishes all three topics at least once."""
        self._wait_for_fresh_data(timeout=timeout)
