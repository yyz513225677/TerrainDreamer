#!/usr/bin/env python3
"""Train the LunarDreamer policy live on the Gazebo / Clearpath J100 sim.

Architecture:
    * Single rclpy node subscribes to lidar / IMU / odom on /j100_0001/*
    * 10 Hz control timer builds an observation, queries the policy, applies
      the SafetyShield, and publishes a TwistStamped on /j100_0001/cmd_vel.
    * Episodes are managed by an internal MissionManager (from
      lunar_dreamer_algorithm) so this script does not depend on the existing
      mission_manager_node — the Dreamer drives, the FSM here decides when
      to switch outbound/return and when to reset.
    * Transitions go into a ReplayBuffer.  A worker thread runs DreamerTrainer
      steps as fast as the buffer allows, so the control loop stays smooth.

Run after sourcing ROS Jazzy + the project workspace:

    cd /home/rickslab3/Documents/Leo/terrain_dreamer
    python3 scripts/train_lunar_dreamer_ros.py
"""
from __future__ import annotations

import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

# Make the algorithm package importable when running from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import rclpy                                       # noqa: E402
from rclpy.node import Node                        # noqa: E402
from rclpy.qos import (                            # noqa: E402
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from geometry_msgs.msg import TwistStamped         # noqa: E402
from nav_msgs.msg import Odometry                  # noqa: E402
from sensor_msgs.msg import Imu, PointCloud2       # noqa: E402
import sensor_msgs_py.point_cloud2 as pc2          # noqa: E402

from lunar_dreamer_algorithm import (              # noqa: E402
    Config,
    DreamerTrainer,
    LunarDreamer,
    MissionManager,
    MissionPhase,
    ReplayBuffer,
    SafetyShield,
    Transition,
    build_goal_vec,
    build_imu_vec,
    pack_obs,
    points_to_bev,
    quat_to_rpy,
)


# ────────────────────────────────────────────────────────────────────────────
# Config knobs specific to the training driver
# ────────────────────────────────────────────────────────────────────────────
CONTROL_HZ = 10.0
EPISODE_TIMEOUT_S = 90.0
TILT_FAIL_RAD = 1.0                # ~57° tilt → end episode as collision
DEST_RADIUS_MIN_M = 4.0
DEST_RADIUS_MAX_M = 10.0
CHECKPOINT_INTERVAL_S = 120.0
LOG_INTERVAL_S = 5.0

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_auto" / "lunar_dreamer_ros"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Gazebo world name (where the flags get spawned).
GAZEBO_WORLD = "lunar_south_pole"
# DEM surface ~3-4 m; place flag base at z=4 so it's visible above terrain.
FLAG_Z_M = 4.0
# Flag SDF kept on one line — protobuf text-format strings can't carry newlines.
FLAG_SDF_TEMPLATE = (
    '<?xml version="1.0"?>'
    '<sdf version="1.10">'
    '<model name="{name}"><static>true</static><link name="link">'
    '<visual name="pole"><pose>0 0 1.0 0 0 0</pose>'
    '<geometry><cylinder><radius>0.04</radius><length>2.0</length></cylinder></geometry>'
    '<material><ambient>0.95 0.95 0.95 1</ambient><diffuse>0.95 0.95 0.95 1</diffuse></material>'
    '</visual>'
    '<visual name="cloth"><pose>0.30 0 1.65 0 0 0</pose>'
    '<geometry><box><size>0.60 0.02 0.40</size></box></geometry>'
    '<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>'
    '<emissive>{er} {eg} {eb} 1</emissive></material>'
    '</visual>'
    '<visual name="base"><pose>0 0 0.01 0 0 0</pose>'
    '<geometry><cylinder><radius>0.25</radius><length>0.02</length></cylinder></geometry>'
    '<material><ambient>0.20 0.20 0.20 1</ambient></material>'
    '</visual>'
    '</link></model></sdf>'
)
YELLOW_RGB = (1.0, 0.9, 0.1)  # outbound (destination) flag
RED_RGB    = (1.0, 0.1, 0.1)  # return (origin) flag


# ────────────────────────────────────────────────────────────────────────────
class DreamerTrainNode(Node):
    def __init__(self):
        super().__init__("dreamer_train_node")

        # ── Algorithm setup ────────────────────────────────────────────────
        self.cfg = Config()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg.device = device
        self.get_logger().info(f"DreamerTrainer device = {device}")

        self.model = LunarDreamer(self.cfg)
        self.trainer = DreamerTrainer(self.model, self.cfg)
        self.buffer = ReplayBuffer(capacity=self.cfg.train.replay_capacity)
        self.shield = SafetyShield(
            self.cfg.safety,
            self.cfg.model.action_low,
            self.cfg.model.action_high,
        )
        self.rssm_state = None

        # ── Latest message buffers (overwritten by callbacks) ──────────────
        self._latest_points: Optional[np.ndarray] = None
        self._latest_imu: Optional[Imu] = None
        self._latest_odom: Optional[Odometry] = None
        self._msg_lock = threading.Lock()

        # ── Episode state ──────────────────────────────────────────────────
        self.mission: Optional[MissionManager] = None
        self.episode_steps = 0
        self.episode_count = 0
        self.last_obs = None
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.last_distance_to_target: Optional[float] = None
        self.total_env_steps = 0
        self.total_train_steps = 0

        # ── Gazebo flag tracking ────────────────────────────────────────────
        # Single active flag per episode; we swap yellow→red on outbound→return.
        self._active_flag_name: Optional[str] = None
        self._last_phase: Optional[MissionPhase] = None

        # ── Subs / pubs ────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.create_subscription(
            PointCloud2, "/j100_0001/sensors/lidar3d_0/points",
            self._on_points, sensor_qos)
        self.create_subscription(
            Imu, "/j100_0001/sensors/imu_0/data",
            self._on_imu, sensor_qos)
        self.create_subscription(
            Odometry, "/j100_0001/platform/odom",
            self._on_odom, reliable_qos)

        self.cmd_pub = self.create_publisher(
            TwistStamped, "/j100_0001/cmd_vel", 10)

        # ── Timers ─────────────────────────────────────────────────────────
        self._dt = 1.0 / CONTROL_HZ
        self.create_timer(self._dt, self._control_step)
        self.create_timer(LOG_INTERVAL_S, self._log)
        self._last_checkpoint = time.time()

        # ── Training worker thread ─────────────────────────────────────────
        self._train_stop = threading.Event()
        self._metrics_lock = threading.Lock()
        self._latest_metrics = {}
        self._train_thread = threading.Thread(
            target=self._train_loop, name="dreamer-train", daemon=True)
        self._train_thread.start()

        self.get_logger().info(
            "DreamerTrainNode ready; waiting for sensors before first action.")

    # ──────────────────────────────────────────────────────────────────────
    # Subscriber callbacks
    # ──────────────────────────────────────────────────────────────────────
    def _on_points(self, msg: PointCloud2) -> None:
        # pc2.read_points returns a numpy *structured* array; extract x/y/z
        # columns and stack into a plain (N, 3) float32.
        try:
            struct = pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)
        except Exception as exc:
            self.get_logger().warn(f"pointcloud parse failed: {exc}")
            return
        if struct is None or len(struct) == 0:
            arr = np.zeros((0, 3), dtype=np.float32)
        else:
            arr = np.stack(
                [struct["x"], struct["y"], struct["z"]], axis=-1
            ).astype(np.float32, copy=False)
        with self._msg_lock:
            self._latest_points = arr

    def _on_imu(self, msg: Imu) -> None:
        with self._msg_lock:
            self._latest_imu = msg

    def _on_odom(self, msg: Odometry) -> None:
        with self._msg_lock:
            self._latest_odom = msg

    # ──────────────────────────────────────────────────────────────────────
    # Snapshot helpers
    # ──────────────────────────────────────────────────────────────────────
    def _snapshot(self):
        with self._msg_lock:
            return (self._latest_points, self._latest_imu, self._latest_odom)

    # ──────────────────────────────────────────────────────────────────────
    # Ground-truth rover pose (used to anchor flags in the gz world frame
    # so they don't end up at odom-drift coords outside the DEM).
    # ──────────────────────────────────────────────────────────────────────
    _GZ_POSE_RE = re.compile(
        r'name:\s*"j100_0001/robot".*?position\s*\{\s*'
        r'x:\s*([-0-9.eE+]+)\s*y:\s*([-0-9.eE+]+)',
        re.DOTALL,
    )

    def _query_rover_gt(self) -> Optional[Tuple[float, float]]:
        try:
            r = subprocess.run(
                ["gz", "topic", "-e",
                 "-t", f"/world/{GAZEBO_WORLD}/dynamic_pose/info",
                 "-n", "1"],
                capture_output=True, text=True, timeout=2.0)
        except Exception:
            return None
        m = self._GZ_POSE_RE.search(r.stdout)
        if not m:
            return None
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            return None

    # ──────────────────────────────────────────────────────────────────────
    # Gazebo flag spawn/remove (visible destination + origin markers)
    # ──────────────────────────────────────────────────────────────────────
    def _gz_spawn_flag(self, name: str, x: float, y: float,
                       rgb: Tuple[float, float, float]) -> bool:
        r, g, b = rgb
        sdf = FLAG_SDF_TEMPLATE.format(
            name=name, r=r, g=g, b=b,
            er=r * 0.25, eg=g * 0.25, eb=b * 0.25,
        )
        req = (
            f"sdf: '{sdf}' "
            f'name: "{name}" '
            f"allow_renaming: true "
            f"pose {{ position {{ x: {x} y: {y} z: {FLAG_Z_M} }} "
            f"orientation {{ w: 1 }} }}"
        )
        try:
            r_ = subprocess.run(
                ["gz", "service", "-s", f"/world/{GAZEBO_WORLD}/create",
                 "--reqtype", "gz.msgs.EntityFactory",
                 "--reptype", "gz.msgs.Boolean",
                 "--timeout", "2000", "--req", req],
                capture_output=True, text=True, timeout=5)
            return "data: true" in r_.stdout
        except Exception as exc:
            self.get_logger().warn(f"[flag] spawn '{name}' failed: {exc}")
            return False

    def _gz_remove_entity(self, name: str) -> bool:
        try:
            r_ = subprocess.run(
                ["gz", "service", "-s", f"/world/{GAZEBO_WORLD}/remove",
                 "--reqtype", "gz.msgs.Entity",
                 "--reptype", "gz.msgs.Boolean",
                 "--timeout", "2000",
                 "--req", f'name: "{name}" type: MODEL'],
                capture_output=True, text=True, timeout=5)
            return "data: true" in r_.stdout
        except Exception:
            return False

    def _remove_active_flag(self) -> None:
        if self._active_flag_name:
            self._gz_remove_entity(self._active_flag_name)
            self._active_flag_name = None

    # ──────────────────────────────────────────────────────────────────────
    # Mission / reset
    # ──────────────────────────────────────────────────────────────────────
    def _sample_destination(self, origin_xy):
        r = random.uniform(DEST_RADIUS_MIN_M, DEST_RADIUS_MAX_M)
        theta = random.uniform(-math.pi, math.pi)
        return (origin_xy[0] + r * math.cos(theta),
                origin_xy[1] + r * math.sin(theta))

    def _reset_episode(self, origin_xy):
        # Clear any leftover flag from the previous episode.
        self._remove_active_flag()

        dest = self._sample_destination(origin_xy)
        self.mission = MissionManager(
            destination=dest,
            origin=origin_xy,
            destination_reach_m=self.cfg.mission.destination_reach_m,
            origin_reach_m=self.cfg.mission.origin_reach_m,
            waypoint_spacing_m=self.cfg.mission.waypoint_spacing_m,
            waypoint_reach_m=self.cfg.mission.waypoint_reach_m,
        )
        self.episode_steps = 0
        self.episode_count += 1
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.last_obs = None
        self.last_distance_to_target = None
        self.rssm_state = None
        self._last_phase = MissionPhase.OUTBOUND

        # Spawn yellow flag at the outbound destination so the rover (and the
        # viewer) can see where it's heading. Odom drifts a lot on heightmap
        # terrain, so we anchor the flag to the rover's *ground-truth* pose
        # plus the local offset (dest - origin); that keeps the flag inside
        # the DEM and right next to the actual rover regardless of drift.
        flag_name = f"dreamer_goal_out_{self.episode_count}"
        local_dx = dest[0] - origin_xy[0]
        local_dy = dest[1] - origin_xy[1]
        gt = self._query_rover_gt()
        if gt is None:
            flag_x, flag_y = dest[0], dest[1]   # fallback — best effort
        else:
            flag_x, flag_y = gt[0] + local_dx, gt[1] + local_dy
        if self._gz_spawn_flag(flag_name, flag_x, flag_y, YELLOW_RGB):
            self._active_flag_name = flag_name
            self.get_logger().info(
                f"[flag] yellow {flag_name} @ "
                f"({flag_x:.2f}, {flag_y:.2f})  (offset {local_dx:+.2f}, {local_dy:+.2f})")

        self.get_logger().info(
            f"=== Episode {self.episode_count}: origin={origin_xy} "
            f"dest=({dest[0]:.2f}, {dest[1]:.2f}) ===")

    def _on_phase_transition(self, new_phase: MissionPhase) -> None:
        """Yellow → red when the rover touches the destination."""
        if new_phase == MissionPhase.RETURN:
            # Remove the yellow destination flag, spawn a red one at origin.
            # Origin is in odom frame; the rover (ground truth) is "right
            # here" so the red flag should appear at the rover's current GT
            # pose plus the (origin - dest) offset = 0 - dest_offset.
            self._remove_active_flag()
            if self.mission is None:
                return
            gt = self._query_rover_gt()
            if gt is None:
                # Fallback: use odom-frame origin directly.
                fx, fy = self.mission.origin
            else:
                # The rover *is* at the destination in odom frame; the origin
                # is `dest - dest_offset`. Project that into GT frame.
                ox, oy = self.mission.origin
                dx_dest, dy_dest = self.mission.destination
                # Local offset from current rover (= dest_odom) to origin.
                fx = gt[0] + (ox - dx_dest)
                fy = gt[1] + (oy - dy_dest)
            flag_name = f"dreamer_goal_home_{self.episode_count}"
            if self._gz_spawn_flag(flag_name, fx, fy, RED_RGB):
                self._active_flag_name = flag_name
                self.get_logger().info(
                    f"[flag] yellow removed → red {flag_name} @ "
                    f"({fx:.2f}, {fy:.2f})  (rover reached destination)")

    # ──────────────────────────────────────────────────────────────────────
    # Observation construction
    # ──────────────────────────────────────────────────────────────────────
    def _build_observation(self, points, imu_msg, odom_msg):
        # Rover pose
        px = odom_msg.pose.pose.position.x
        py = odom_msg.pose.pose.position.y
        q = odom_msg.pose.pose.orientation
        roll, pitch, yaw = quat_to_rpy(q.x, q.y, q.z, q.w)

        # On the very first observation of a run, anchor the origin here so
        # the rover doesn't keep chasing some long-dead (0, 0).
        if self.mission is None:
            self._reset_episode((px, py))

        # Mission FSM step → current target / phase.
        status = self.mission.update(px, py)
        if status.phase != self._last_phase:
            self._on_phase_transition(status.phase)
            self._last_phase = status.phase

        # BEV
        bev = points_to_bev(
            points, grid_size=self.cfg.model.bev_shape[1], extent_m=15.0)

        # IMU vec — pull body velocity from odom (twist is in body frame).
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

        goal_vec = build_goal_vec((px, py), yaw, status.target)
        phase = self.mission.phase_one_hot()
        obs = pack_obs(bev, imu_vec, goal_vec, phase, self.prev_action)

        info = {
            "rover_xy": (px, py),
            "yaw": yaw,
            "tilt": math.hypot(roll, pitch),
            "phase": status.phase,
            "distance_to_target": status.distance_to_target,
            "target": status.target,
        }
        return obs, info

    # ──────────────────────────────────────────────────────────────────────
    # Reward + termination
    # ──────────────────────────────────────────────────────────────────────
    def _reward_and_done(self, info):
        rew = self.cfg.reward.step_penalty
        if self.last_distance_to_target is not None:
            progress = self.last_distance_to_target - info["distance_to_target"]
            rew += self.cfg.reward.progress_weight * progress
        rew += self.cfg.reward.angular_penalty * abs(float(self.prev_action[1]))

        done = False
        if info["tilt"] > TILT_FAIL_RAD:
            rew += self.cfg.reward.collision_penalty
            done = True
        if self.mission.is_done():
            rew += self.cfg.reward.success_bonus
            done = True
        if self.episode_steps >= int(EPISODE_TIMEOUT_S * CONTROL_HZ):
            done = True
        return float(rew), bool(done)

    # ──────────────────────────────────────────────────────────────────────
    # Action publication
    # ──────────────────────────────────────────────────────────────────────
    def _publish_cmd(self, action_np):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(action_np[0])
        msg.twist.angular.z = float(action_np[1])
        self.cmd_pub.publish(msg)

    def _publish_stop(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.cmd_pub.publish(msg)

    # ──────────────────────────────────────────────────────────────────────
    # Control loop (10 Hz)
    # ──────────────────────────────────────────────────────────────────────
    def _control_step(self):
        points, imu_msg, odom_msg = self._snapshot()
        if points is None or imu_msg is None or odom_msg is None:
            return     # waiting for sensors

        obs, info = self._build_observation(points, imu_msg, odom_msg)

        # Policy step (no grad; on training device).
        device = next(self.model.parameters()).device
        gpu_obs = {k: v.unsqueeze(0).to(device) for k, v in obs.items()}
        with torch.no_grad():
            action_t, self.rssm_state = self.model.act(
                gpu_obs, prev_state=self.rssm_state, prev_action=None,
                deterministic=False)
            # Safety shield with a forward-cone lidar slice carved from points.
            forward_min = self._forward_min_from_points(points, device)
            shielded = self.shield.apply(
                action_t, forward_min=forward_min.unsqueeze(0))
        action_np = shielded[0].detach().cpu().numpy()

        # Reward / done bookkeeping using the *current* observation as
        # next-state of the previous transition.
        if self.last_obs is not None:
            reward, done = self._reward_and_done(info)
            transition = Transition(
                obs=self.last_obs,
                action=torch.from_numpy(self.prev_action.copy()),
                reward=reward,
                done=done,
                next_obs=obs,
                info={
                    "phase": info["phase"].name,
                    "distance_to_target": info["distance_to_target"],
                },
            )
            self.buffer.add(transition)
            self.total_env_steps += 1

            if done:
                self._publish_stop()
                self._reset_episode(info["rover_xy"])
                # Update last_distance_to_target after reset so the next step
                # measures progress from the new origin.
                return

        # Publish action and roll bookkeeping forward.
        self._publish_cmd(action_np)
        self.prev_action = action_np.astype(np.float32)
        self.last_obs = obs
        self.last_distance_to_target = info["distance_to_target"]
        self.episode_steps += 1

        # Checkpoint occasionally.
        now = time.time()
        if now - self._last_checkpoint > CHECKPOINT_INTERVAL_S:
            self._save_checkpoint()
            self._last_checkpoint = now

    # ──────────────────────────────────────────────────────────────────────
    # Forward-cone min distance from raw points (no separate scan topic)
    # ──────────────────────────────────────────────────────────────────────
    def _forward_min_from_points(self, points: np.ndarray, device) -> torch.Tensor:
        if points.size == 0:
            return torch.tensor(float("inf"), device=device)
        forward_arc = 0.7   # rad: |atan2(y, x)| < arc
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        mask = (
            (x > 0.2) & (np.abs(y) < (np.abs(x) * math.tan(forward_arc))) &
            (z > -0.3) & (z < 1.0)
        )
        if not mask.any():
            return torch.tensor(float("inf"), device=device)
        r = np.hypot(x[mask], y[mask]).min()
        return torch.tensor(float(r), device=device)

    # ──────────────────────────────────────────────────────────────────────
    # Training worker
    # ──────────────────────────────────────────────────────────────────────
    def _train_loop(self):
        torch.set_num_threads(1)        # keep CPU side cheap
        while not self._train_stop.is_set():
            if not self.buffer.can_sample(
                self.cfg.train.batch_size, self.cfg.train.seq_len
            ) or self.total_env_steps < self.cfg.train.min_replay:
                time.sleep(0.5)
                continue
            try:
                metrics = self.trainer.step(self.buffer)
            except Exception as exc:
                self.get_logger().warn(f"trainer.step failed: {exc}")
                time.sleep(1.0)
                continue
            if metrics is None:
                time.sleep(0.1)
                continue
            self.total_train_steps += 1
            with self._metrics_lock:
                self._latest_metrics = metrics
            # Tiny breather so the GIL doesn't starve the rclpy thread.
            time.sleep(0.02)

    # ──────────────────────────────────────────────────────────────────────
    # Periodic logging
    # ──────────────────────────────────────────────────────────────────────
    def _log(self):
        with self._metrics_lock:
            metrics = dict(self._latest_metrics)
        msg = (
            f"env_steps={self.total_env_steps} "
            f"train_steps={self.total_train_steps} "
            f"episodes={self.episode_count} "
            f"buffer={len(self.buffer)} "
        )
        if metrics:
            msg += " | " + " ".join(
                f"{k}={v:.3f}" for k, v in metrics.items())
        self.get_logger().info(msg)

    # ──────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────────────
    def _save_checkpoint(self, keep_n: int = 5):
        path = CHECKPOINT_DIR / f"dreamer_step_{self.total_train_steps:07d}.pt"
        try:
            self.trainer.save(str(path))
            self.get_logger().info(f"checkpoint -> {path}")
        except Exception as exc:
            self.get_logger().warn(f"checkpoint save failed: {exc}")
            return
        try:
            ckpts = sorted(CHECKPOINT_DIR.glob("dreamer_step_*.pt"))
            for old in ckpts[:-keep_n]:
                old.unlink(missing_ok=True)
        except Exception as exc:
            self.get_logger().warn(f"checkpoint cleanup failed: {exc}")

    # ──────────────────────────────────────────────────────────────────────
    def shutdown(self):
        self._train_stop.set()
        self._publish_stop()
        self._remove_active_flag()
        try:
            self._save_checkpoint()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = DreamerTrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
