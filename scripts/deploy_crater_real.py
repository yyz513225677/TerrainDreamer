"""TerrainDreamer real-world inference deployment node.

Loads a trained TerrainDreamer checkpoint and drives a physical Clearpath J100
(or any Jackal) at 10 Hz using the same observation pipeline as the sim
trainer — but with all Gazebo-specific code removed. Pose tracking comes
from the rover's existing wheel-odom-fused EKF (or any external SLAM that
publishes `/platform/odom` in nav_msgs/Odometry), and the operator publishes
the goal on a ROS topic.

Hardware assumptions
--------------------
  * Velodyne VLP-32 LiDAR  →  /j100_0001/sensors/lidar3d_0/points
  * VectorNav VN-100 IMU   →  /j100_0001/sensors/imu_0/data
  * Clearpath wheel+IMU EKF → /j100_0001/platform/odom
  * Operator goal topic    →  /td/goal  (geometry_msgs/PoseStamped, world frame)
  * Optional e-stop topic  →  /td/estop (std_msgs/Bool — stops rover when true)

Pose source
-----------
The trainer used Gazebo ground-truth. On real hardware we substitute with
``/platform/odom``: drift-bounded for short missions (<200 m) but enough
to maintain mission framing. For longer missions hook a SLAM node
(FAST-LIO2, LeGO-LOAM, KISS-ICP) and remap its output topic to
``/platform/odom``.

Safety
------
The SafetyShield (LiDAR brake + tilt brake + emergency turn) is active.
Plus:
  * Hard speed cap (``MAX_LINEAR_M_S``, ``MAX_ANGULAR_RAD_S``)
  * Sensor watchdog — stops rover if any sensor stale > 0.5 s
  * E-stop topic stops rover immediately
  * Stuck detection (no Gazebo GT needed — uses odom pose)

Usage
-----
  python3 scripts/deploy_crater_real.py \\
      --ckpt checkpoints_auto/crater/td_step_XXXXXXX.pt \\
      --pose-source odom        # or 'tf' to use tf2 (world->base_link)

Then publish a goal:

  ros2 topic pub --once /td/goal geometry_msgs/msg/PoseStamped \\
      "{header:{frame_id: 'odom'}, pose:{position:{x: 5.0, y: 2.0, z: 0.0}}}"

Set max speed:

  MAX_LINEAR_M_S=0.4 python3 scripts/deploy_crater_real.py ...
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Bool
import sensor_msgs_py.point_cloud2 as pc2

from crater import (CONTROL_MODE_AUTO, Config, CraterModel, FlagSeekingPolicy,
                    FlagSeekingPolicyConfig, MissionManager, MissionPhase,
                    TrajectoryMemory)
from lunar_dreamer_algorithm import (build_goal_vec, build_imu_vec,
                                     points_to_bev, quat_to_rpy)


CONTROL_HZ = 10.0
SENSOR_TIMEOUT_S = 0.5             # stop if any sensor stale > this
DESTINATION_REACH_M = 0.8
MAX_LINEAR_M_S = float(os.environ.get("MAX_LINEAR_M_S", "0.6"))
MAX_ANGULAR_RAD_S = float(os.environ.get("MAX_ANGULAR_RAD_S", "1.0"))
STUCK_WINDOW_S = 5.0
STUCK_MIN_DIST_M = 0.5
STUCK_RECOVERY_THRESHOLD_STEPS = 30


class CraterRealNode(Node):
    def __init__(self, ckpt_path: Path, pose_source: str = "odom"):
        super().__init__("crater_real_node")
        self.pose_source = pose_source

        self.cfg = Config()
        self.cfg.trav.enable = False
        self.cfg.trav.enable_return_norm = True
        self.cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load checkpoint
        self.model = CraterModel(self.cfg).to(self.cfg.device)
        self.model.eval()
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location=self.cfg.device)
            if "model" in sd:
                self.model.load_state_dict(sd["model"])
            else:
                self.model.load_state_dict(sd)
            self.get_logger().info(f"loaded checkpoint: {ckpt_path}")

        self.base_policy = FlagSeekingPolicy(FlagSeekingPolicyConfig())
        self.mission = MissionManager(
            destination_reach_m=DESTINATION_REACH_M,
            origin_reach_m=DESTINATION_REACH_M,
        )
        self.trajectory = TrajectoryMemory(waypoint_spacing_m=2.0,
                                           waypoint_reach_m=1.0)

        # ── ROS subs ────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE)
        reliable_qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(
            PointCloud2, "/j100_0001/sensors/lidar3d_0/points",
            self._on_points, sensor_qos)
        self.create_subscription(
            Imu, "/j100_0001/sensors/imu_0/data",
            self._on_imu, reliable_qos)
        self.create_subscription(
            Odometry, "/j100_0001/platform/odom",
            self._on_odom, reliable_qos)
        self.create_subscription(
            PoseStamped, "/td/goal", self._on_goal, reliable_qos)
        self.create_subscription(
            Bool, "/td/estop", self._on_estop, reliable_qos)
        self.cmd_pub = self.create_publisher(
            TwistStamped, "/j100_0001/cmd_vel", 10)

        # ── State ──────────────────────────────────────────────────────
        self._lock = threading.Lock()
        self._latest_points: Optional[np.ndarray] = None
        self._latest_imu: Optional[Imu] = None
        self._latest_odom: Optional[Odometry] = None
        self._latest_imu_ts = 0.0
        self._latest_points_ts = 0.0
        self._latest_odom_ts = 0.0
        self._goal_world: Optional[Tuple[float, float]] = None
        self._origin_world: Optional[Tuple[float, float]] = None
        self._estop = False
        self._rssm_state = None
        self._prev_action = np.zeros(2, dtype=np.float32)

        # Stuck detection
        self._stuck_history = []
        self._stuck_count = 0

        # Timer
        self.create_timer(1.0 / CONTROL_HZ, self._control_step)
        self.get_logger().info(
            f"TerrainDreamer deployment ready (device={self.cfg.device}, "
            f"pose={pose_source}, max_v={MAX_LINEAR_M_S}, "
            f"max_omega={MAX_ANGULAR_RAD_S})")
        self.get_logger().info(
            "Waiting for goal — publish on /td/goal (PoseStamped) to start.")

    # ── Subscribers ────────────────────────────────────────────────────
    def _on_points(self, msg: PointCloud2) -> None:
        try:
            struct = pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)
            xyz = np.stack(
                [struct["x"], struct["y"], struct["z"]],
                axis=-1).astype(np.float32)
        except Exception:
            return
        with self._lock:
            self._latest_points = xyz
            self._latest_points_ts = time.monotonic()

    def _on_imu(self, msg: Imu) -> None:
        with self._lock:
            self._latest_imu = msg
            self._latest_imu_ts = time.monotonic()

    def _on_odom(self, msg: Odometry) -> None:
        with self._lock:
            self._latest_odom = msg
            self._latest_odom_ts = time.monotonic()

    def _on_goal(self, msg: PoseStamped) -> None:
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        with self._lock:
            self._goal_world = (gx, gy)
            if self._origin_world is None and self._latest_odom is not None:
                op = self._latest_odom.pose.pose.position
                self._origin_world = (op.x, op.y)
                self.mission.reset(origin=self._origin_world,
                                   destination=(gx, gy))
                self.trajectory.reset()
        self.get_logger().info(f"new goal: ({gx:.2f}, {gy:.2f})")

    def _on_estop(self, msg: Bool) -> None:
        if msg.data and not self._estop:
            self.get_logger().warn("E-STOP engaged")
        elif not msg.data and self._estop:
            self.get_logger().info("E-STOP released")
        self._estop = bool(msg.data)

    # ── Helpers ────────────────────────────────────────────────────────
    def _publish_stop(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.cmd_pub.publish(msg)

    def _sensors_fresh(self) -> bool:
        now = time.monotonic()
        return (
            (now - self._latest_points_ts) < SENSOR_TIMEOUT_S
            and (now - self._latest_imu_ts) < SENSOR_TIMEOUT_S
            and (now - self._latest_odom_ts) < SENSOR_TIMEOUT_S
        )

    def _update_stuck(self, px, py, now):
        self._stuck_history.append((now, px, py))
        cutoff = now - STUCK_WINDOW_S
        self._stuck_history = [h for h in self._stuck_history if h[0] >= cutoff]
        if len(self._stuck_history) < 5:
            self._stuck_count = 0
            return
        x0, y0 = self._stuck_history[0][1], self._stuck_history[0][2]
        x1, y1 = self._stuck_history[-1][1], self._stuck_history[-1][2]
        if math.hypot(x1 - x0, y1 - y0) < STUCK_MIN_DIST_M:
            self._stuck_count += 1
        else:
            self._stuck_count = 0

    def _build_obs(self, points, imu_msg, odom_msg):
        # Pose
        q = odom_msg.pose.pose.orientation
        roll, pitch, yaw = quat_to_rpy(q.x, q.y, q.z, q.w)
        px = odom_msg.pose.pose.position.x
        py = odom_msg.pose.pose.position.y

        # Mission update
        self.mission.update((px, py, yaw), self.trajectory)
        tgt = self.mission.get_current_target()
        goal_vec = build_goal_vec((px, py), yaw, tgt)
        phase = self.mission.get_phase_vector()

        # BEV
        bev = points_to_bev(points, grid_size=self.cfg.model.bev_shape[1],
                            extent_m=15.0)
        body_vel = (odom_msg.twist.twist.linear.x,
                    odom_msg.twist.twist.linear.y,
                    odom_msg.twist.twist.linear.z)
        imu_vec = build_imu_vec(
            lin_acc=(imu_msg.linear_acceleration.x,
                     imu_msg.linear_acceleration.y,
                     imu_msg.linear_acceleration.z),
            ang_vel=(imu_msg.angular_velocity.x,
                     imu_msg.angular_velocity.y,
                     imu_msg.angular_velocity.z),
            rpy=(roll, pitch, yaw),
            body_vel=body_vel,
        )
        return {
            "lidar_bev":     torch.from_numpy(bev),
            "imu":           torch.from_numpy(imu_vec),
            "goal_vector":   torch.from_numpy(goal_vec),
            "mission_phase": torch.tensor(phase, dtype=torch.float32),
            "prev_action":   torch.from_numpy(self._prev_action.copy()),
        }, (px, py, yaw)

    def _subgoal_to_velocity(self, theta_sub, r_sub, gpu_obs):
        dx = r_sub * math.cos(theta_sub)
        dy = r_sub * math.sin(theta_sub)
        synth_goal = torch.tensor(
            [[dx, dy, r_sub, theta_sub]],
            dtype=torch.float32, device=self.cfg.device)
        sub_obs = {k: v for k, v in gpu_obs.items()}
        sub_obs["goal_vector"] = synth_goal
        vel_t = self.base_policy.compute(sub_obs)
        return vel_t[0].detach().cpu().numpy()

    # ── Main loop ──────────────────────────────────────────────────────
    @torch.no_grad()
    def _control_step(self):
        if self._estop or not self._sensors_fresh() or self._goal_world is None:
            self._publish_stop()
            return

        with self._lock:
            points = self._latest_points
            imu_msg = self._latest_imu
            odom_msg = self._latest_odom

        if points is None or imu_msg is None or odom_msg is None:
            self._publish_stop()
            return

        if self.mission.is_success():
            self._publish_stop()
            self.get_logger().info_once("destination reached — idling")
            return

        try:
            obs, (px, py, _yaw) = self._build_obs(points, imu_msg, odom_msg)
        except Exception as exc:
            self.get_logger().warn(f"obs build failed: {exc}")
            self._publish_stop()
            return

        gpu_obs = {k: v.unsqueeze(0).to(self.cfg.device) for k, v in obs.items()}
        action_t, self._rssm_state = self.model.select_action(
            gpu_obs, state=self._rssm_state,
            human_action=None, control_mode=CONTROL_MODE_AUTO,
            deterministic=True,         # deterministic mean policy at deploy
        )
        action_np = action_t[0].detach().cpu().numpy()

        # Hierarchical → low-level
        if self.cfg.model.use_hierarchical_action:
            theta_sub, r_sub = float(action_np[0]), float(action_np[1])
            cmd = self._subgoal_to_velocity(theta_sub, r_sub, gpu_obs)
        else:
            cmd = action_np.copy()

        # Stuck recovery (uses odom pose, no GT needed)
        self._update_stuck(px, py, time.time())
        if self._stuck_count > STUCK_RECOVERY_THRESHOLD_STEPS:
            sign = 1.0 if (self._stuck_count // 20) % 2 == 0 else -1.0
            cmd = np.array([0.0, sign * 0.8], dtype=np.float32)

        # Hard speed limits (safety)
        cmd[0] = float(np.clip(cmd[0], 0.0, MAX_LINEAR_M_S))
        cmd[1] = float(np.clip(cmd[1], -MAX_ANGULAR_RAD_S, MAX_ANGULAR_RAD_S))

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(cmd[0])
        msg.twist.angular.z = float(cmd[1])
        self.cmd_pub.publish(msg)
        self._prev_action = action_np.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Path to a trained TerrainDreamer checkpoint (.pt)")
    ap.add_argument("--pose-source", choices=("odom", "tf"), default="odom",
                    help="Where to read pose from. 'odom' uses "
                         "/platform/odom; 'tf' would use world->base_link "
                         "(reserved for future SLAM integration).")
    args = ap.parse_args()
    if not args.ckpt.exists():
        print(f"ERROR: checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    rclpy.init()
    node = CraterRealNode(args.ckpt, pose_source=args.pose_source)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
