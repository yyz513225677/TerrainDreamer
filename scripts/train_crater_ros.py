#!/usr/bin/env python3
"""ROS 2 / Gazebo Sim training driver for the **TerrainDreamer** algorithm.

This is the ROS-side Gymnasium-style wrapper that:
  * subscribes to Clearpath J100 sensors (lidar / IMU / odom),
  * builds the observation dict expected by `crater`
    (lidar_bev, imu, goal_vector, mission_phase, prev_action),
  * drives the rover via /j100_0001/cmd_vel,
  * spawns visible flags in gz anchored to the rover's ground-truth pose,
  * runs `Trainer.train_step()` in a worker thread.

The algorithm itself (`crater/`) is ROS-free. This script lives
outside that module because it is, by design, ROS-specific.

Obs-builder utility functions are reused from the older
`lunar_dreamer_algorithm.obs_builders` since they are wrapper-side code
(not algorithm-internal).

Run after sourcing ROS Jazzy + the workspace:
    cd /home/rickslab3/Documents/Leo/terrain_dreamer
    python3 scripts/train_crater_ros.py
"""
from __future__ import annotations

import math
import os
import queue
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import rclpy                                       # noqa: E402
from rclpy.node import Node                        # noqa: E402
from rclpy.qos import (                            # noqa: E402
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy,
)
from geometry_msgs.msg import PoseStamped, TwistStamped  # noqa: E402
from nav_msgs.msg import Odometry                  # noqa: E402
from sensor_msgs.msg import Imu, PointCloud2       # noqa: E402
import sensor_msgs_py.point_cloud2 as pc2          # noqa: E402

from crater import (                      # noqa: E402
    CONTROL_MODE_AUTO,
    CONTROL_MODE_HUMAN,
    Config,
    FlagSeekingPolicy,
    FlagSeekingPolicyConfig,
    MissionManager,
    MissionPhase,
    ReplayBuffer,
    CraterModel,
    Trainer,
    TrajectoryMemory,
    Transition,
)

# Reuse obs-builder utilities from the older module (pure wrapper code).
from lunar_dreamer_algorithm import (              # noqa: E402
    build_goal_vec, build_imu_vec, points_to_bev, quat_to_rpy,
)


# ────────────────────────────────────────────────────────────────────────────
CONTROL_HZ = 10.0
EPISODE_TIMEOUT_S = 300.0
TILT_FAIL_RAD = 0.7
DEST_MIN_DISTANCE_M = 4.0             # minimum distance from origin (avoid trivial goals)
CHECKPOINT_INTERVAL_S = 120.0
LOG_INTERVAL_S = 5.0

# DEM is 100 m × 100 m centred at origin (so ±50 m); sample destinations
# uniformly across the entire DEM with a small safety margin so the flag
# never lands off the heightmap.
DEM_HALF_M = 40.0                     # destination box: x, y ∈ ±40 m
RESPAWN_HEIGHT_M = 4.0                # drop height for teleport-to-origin
RESET_SETTLE_S = 2.0                  # wait for rover to settle after teleport

# Base policy: hand-crafted flag-seeking controller. When enabled, its action
# is routed through HumanTakeover as a "human demonstration" so the BC loss
# trains the learned actor to imitate it. Set USE_BASE_POLICY=0 in the env
# to disable (and run the learned actor directly).
USE_BASE_POLICY = os.environ.get("USE_BASE_POLICY", "1") != "0"
# Human teleop mode: keep ALL sim/flag/mission infrastructure but skip the
# learned actor and feed WASD keyboard input as the cmd. Demos are saved to
# demos/demo_NNN.npz (same schema as scripts/human_drive.py) instead of
# accumulating in the replay buffer.
HUMAN_DRIVE = os.environ.get("HUMAN_DRIVE", "0") == "1"
# Ablation flags. Setting these to "1" disables the corresponding TerrainDreamer
# contribution so we can isolate its effect. With all four = "1" the
# trainer reduces to a DreamerV3 baseline (no interrupt safety filter,
# no demo distillation, no hierarchical reverse, BC anchor optional).
ABLATE_INTERRUPTS    = os.environ.get("ABLATE_INTERRUPTS",    "0") == "1"
ABLATE_DISTILLATION  = os.environ.get("ABLATE_DISTILLATION",  "0") == "1"
ABLATE_DEMO_ANCHOR   = os.environ.get("ABLATE_DEMO_ANCHOR",   "0") == "1"
ABLATE_REVERSE       = os.environ.get("ABLATE_REVERSE",       "0") == "1"
# Which sub-goal demo source to use: simple | reactive | memory
#   simple   — bearing straight at destination (no obstacle awareness)
#   reactive — instantaneous BEV-aware sweep avoidance
#   memory   — reactive + persistent visited-ground memory channel
BASE_POLICY_MODE = os.environ.get("BASE_POLICY_MODE", "simple")

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_auto" / "crater"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

GAZEBO_WORLD = os.environ.get("GZ_WORLD", "lunar_south_pole")
# Optional obstacle mask (8-bit PNG, 0 = traversable, 255 = obstacle).
# Looked up by world name so the default landscape doesn't need one.
OBSTACLE_MASK_PATHS = {
    "lunar_south_pole_rugged": (PROJECT_ROOT / "lunar_south_pole_gazebo"
                                / "data" / "heightmaps"
                                / "rugged_obstacle_mask.png"),
    "lunar_south_pole_extreme": (PROJECT_ROOT / "lunar_south_pole_gazebo"
                                 / "data" / "heightmaps"
                                 / "extreme_obstacle_mask.png"),
}
DEM_FULL_M = 100.0   # heightmap covers ±DEM_FULL_M/2 in world frame
FLAG_Z_M = 0.0   # spawn flags at ground level (was 4m floating above)
FLAG_SDF_TEMPLATE = (
    '<?xml version="1.0"?>'
    '<sdf version="1.10">'
    '<model name="{name}"><static>true</static><link name="link">'
    '<visual name="pole"><pose>0 0 2.0 0 0 0</pose>'
    '<geometry><cylinder><radius>0.05</radius><length>4.0</length></cylinder></geometry>'
    '<material><ambient>0.95 0.95 0.95 1</ambient><diffuse>0.95 0.95 0.95 1</diffuse></material>'
    '</visual>'
    '<visual name="cloth"><pose>0.40 0 3.55 0 0 0</pose>'
    '<geometry><box><size>0.80 0.02 0.50</size></box></geometry>'
    '<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>'
    '<emissive>{er} {eg} {eb} 1</emissive></material>'
    '</visual>'
    '<visual name="base"><pose>0 0 0.01 0 0 0</pose>'
    '<geometry><cylinder><radius>0.25</radius><length>0.02</length></cylinder></geometry>'
    '<material><ambient>0.20 0.20 0.20 1</ambient></material>'
    '</visual>'
    '</link></model></sdf>'
)
YELLOW_RGB = (1.0, 0.9, 0.1)
RED_RGB    = (1.0, 0.1, 0.1)

# ── Boundary walls ────────────────────────────────────────────────────────
# Build a perimeter of static walls around the DEM so the rover physically
# can't drive off the heightmap. Wall top must clear the DEM's highest point
# by a safety margin so the rover can't roll over a high spot. The lunar
# heightmap has z ∈ [0, ~4 m] (SDF <size>100 100 4</size>), so the wall top
# sits at LAND_MAX + TOP_MARGIN and the bottom is buried slightly below the
# lowest terrain so there's no gap underneath.
WALL_LAND_MAX_M = 4.0                # DEM max elevation (from SDF size_z)
WALL_TOP_MARGIN_M = 0.6              # user requested margin above DEM peak
WALL_BOTTOM_Z_M = -0.5               # bury wall base half a metre underground
WALL_HEIGHT_M = WALL_LAND_MAX_M + WALL_TOP_MARGIN_M - WALL_BOTTOM_Z_M  # 5.1
WALL_CENTER_Z_M = WALL_BOTTOM_Z_M + WALL_HEIGHT_M / 2.0                # 2.05
WALL_THICKNESS_M = 0.5
WALL_HALF_M = 50.0                   # DEM is 100 m × 100 m → ±50 m
WALL_LENGTH_M = 2 * WALL_HALF_M + WALL_THICKNESS_M  # overlap at corners
WALL_SDF_TEMPLATE = (
    '<?xml version="1.0"?>'
    '<sdf version="1.10">'
    '<model name="{name}"><static>true</static><link name="link">'
    '<pose>0 0 {hz} 0 0 0</pose>'
    '<collision name="collision">'
    '<geometry><box><size>{sx} {sy} {sz}</size></box></geometry>'
    '</collision>'
    '<visual name="visual">'
    '<geometry><box><size>{sx} {sy} {sz}</size></box></geometry>'
    '<material>'
    '<ambient>0.55 0.25 0.10 1</ambient>'
    '<diffuse>0.65 0.30 0.12 1</diffuse>'
    '<emissive>0.10 0.05 0.02 1</emissive>'
    '</material>'
    '</visual>'
    '</link></model></sdf>'
)


# ────────────────────────────────────────────────────────────────────────────
class CraterTrainNode(Node):
    def __init__(self):
        super().__init__("crater_train_node")

        # ── Algorithm setup ────────────────────────────────────────────────
        self.cfg = Config()
        # ── Demo-anchored regime + hierarchical sub-goal ───────────────────
        # We DISABLE the traversability reward shaping: in the demo-only
        # regime the head has no supervised label and its output is noise,
        # which the adaptive Lagrangian then amplifies into a huge per-step
        # penalty, locking the critic into a large-negative regime.
        # Return normalization stays on (separately) — it's the DreamerV3
        # trick that keeps the actor stable under any reward scale.
        self.cfg.trav.enable = False
        self.cfg.trav.enable_return_norm = True
        # ── Env-var overrides for the auto-improvement loop ───────────────
        # Allows the orchestrator to mutate key hyperparameters across
        # iterations without editing config.py. All are optional; defaults
        # come from the Config dataclasses.
        if "BC_WEIGHT" in os.environ:
            self.cfg.bc.behavior_cloning_weight = float(os.environ["BC_WEIGHT"])
        if "RECOVERY_BC_WEIGHT" in os.environ:
            self.cfg.bc.recovery_behavior_cloning_weight = float(
                os.environ["RECOVERY_BC_WEIGHT"])
        if "DREAMER_ACTOR_WEIGHT" in os.environ:
            self.cfg.bc.dreamer_actor_weight = float(
                os.environ["DREAMER_ACTOR_WEIGHT"])
        # Ablation: ABLATE_DEMO_ANCHOR=1 disables the BC anchor, leaving
        # only DreamerV3 imagined-return loss. This is the vanilla-Dreamer
        # baseline. Matched by raising the dreamer weight to 1.0 so the
        # actor is not undertrained.
        if ABLATE_DEMO_ANCHOR:
            self.cfg.bc.behavior_cloning_weight = 0.0
            self.cfg.bc.recovery_behavior_cloning_weight = 0.0
            self.cfg.bc.dreamer_actor_weight = 1.0
            self.get_logger().info(
                "[ablation] DEMO_ANCHOR disabled: BC_WEIGHT=0, "
                "DREAMER_ACTOR_WEIGHT=1.0")
        if "TIMEOUT_PENALTY" in os.environ:
            self.cfg.reward.timeout_penalty = float(
                os.environ["TIMEOUT_PENALTY"])
        if "SUB_R_MAX" in os.environ:
            r_max = float(os.environ["SUB_R_MAX"])
            self.cfg.model.action_high = (math.pi, r_max)
        # USE_HIERARCHICAL_ACTION=0 forces flat [v, omega] action space —
        # used for the vanilla-DreamerV3 baseline (paper Section 5.3).
        if os.environ.get("USE_HIERARCHICAL_ACTION", "1") == "0":
            self.cfg.model.use_hierarchical_action = False
            self.cfg.model.action_low = (0.0, -1.0)
            self.cfg.model.action_high = (0.8, 1.0)
        # Seed control for paper reproducibility (SEED env var).
        if "SEED" in os.environ:
            seed = int(os.environ["SEED"])
            import random as _random
            _random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.get_logger().info(f"SEED={seed} applied (random/numpy/torch).")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg.device = device
        self.get_logger().info(
            f"TerrainDreamer device={device}  "
            f"BC_W={self.cfg.bc.behavior_cloning_weight}  "
            f"DREAMER_AW={self.cfg.bc.dreamer_actor_weight}  "
            f"BASE_MODE={BASE_POLICY_MODE}  "
            f"R_MAX={self.cfg.model.action_high[1]}")

        self.model = CraterModel(self.cfg)
        self.trainer = Trainer(self.model, self.cfg)
        self.buffer = ReplayBuffer(capacity=self.cfg.train.replay_capacity)
        self.rssm_state = None

        # Mission FSM lives outside the model; the wrapper drives reset()
        # whenever the episode timeout or destination is reached.
        self.mission = MissionManager(
            destination_reach_m=self.cfg.mission.destination_reach_m,
            origin_reach_m=self.cfg.mission.origin_reach_m,
        )
        self.trajectory = TrajectoryMemory(
            waypoint_spacing_m=self.cfg.mission.waypoint_spacing_m,
            waypoint_reach_m=self.cfg.mission.waypoint_reach_m,
        )

        # ── Optional obstacle mask (per-world) ─────────────────────────
        # Loaded from rugged_obstacle_mask.png (8-bit, 0=traversable,
        # 255=obstacle). When present, _sample_destination_in_world rejects
        # any point landing on an obstacle pixel so the flag is always
        # reachable.
        self._obstacle_mask = None
        mask_path = OBSTACLE_MASK_PATHS.get(GAZEBO_WORLD)
        if mask_path is not None and mask_path.exists():
            try:
                from PIL import Image as _PI
                arr = np.array(_PI.open(mask_path).convert("L"))
                self._obstacle_mask = arr > 127
                self.get_logger().info(
                    f"[obstacle] loaded mask {mask_path.name} "
                    f"({arr.shape[0]}x{arr.shape[1]}, "
                    f"{100.0 * self._obstacle_mask.mean():.1f}% blocked)")
            except Exception as exc:
                self.get_logger().warn(
                    f"[obstacle] failed to load mask: {exc}")

        # Flag-seeking base policy. Acts as a demonstration source: its
        # action is routed through HumanTakeover so the trainer's BC loss
        # trains the learned actor to imitate it.
        # ── Stuck detection: track recent GT positions and goal distance.
        # Two trigger paths:
        #  (a) tight bbox — rover stays inside a small region (still useful
        #      for sideways/rotating motion that doesn't help goal progress)
        #  (b) no goal progress — distance-to-target hasn't decreased
        #      meaningfully over the window. This catches the "small
        #      oscillation 0.6 m while struggling on a slope" case.
        self._stuck_window_s = 5.0
        self._stuck_min_dist_m = 0.7         # bbox threshold
        self._stuck_min_progress_m = 0.3     # need ≥0.3 m progress toward goal
        self._stuck_history = []  # list of (ts, x, y, distance_to_target)
        self._stuck_count = 0
        self._stuck_rotation_target_yaw = None
        # Recovery direction is locked per stuck event so the rover commits
        # to one side instead of flipping every 2 s.
        self._stuck_recovery_dir = 0.0
        # Scripted recovery maneuver. Once triggered, runs as a fixed
        # phase sequence and CANNOT be retriggered by stuck detection
        # (otherwise progress_stuck stays true during pure rotation and
        # the recovery feeds itself forever). Sequence (10 Hz):
        #   phase A: reverse + small rotate, 10 steps (1 s)
        #   phase B: pure rotate ±0.8, 20 steps (2 s ≈ 90°)
        #   phase C: drive forward, 35 steps (3.5 s) — actually translate
        # Total 65 steps / 6.5 s. After phase C we enter a 50-step (5 s)
        # "cooldown" window where stuck detection cannot re-arm; this
        # gives BASE time to make goal progress.
        self._recovery_phase = 0        # 0 = inactive, 1=rev, 2=rot, 3=fwd, 4=cooldown
        self._recovery_step = 0
        # Post-stuck commit: after escaping a stuck zone, drive straight
        # ahead for this many control steps before re-engaging goal-seeking
        # — prevents the rover from immediately returning to the same
        # un-climbable obstacle that caused the stuck.
        self._post_stuck_commit_steps = 0
        self._post_stuck_commit_duration = 30   # 3 s at 10 Hz
        # Distance to target at the moment the stuck/slope event triggered.
        # The post-X forward commit aborts as soon as the rover has made
        # net progress past that distance — i.e. once we're closer to the
        # goal than when we got stuck, we've crossed the obstacle and the
        # commit isn't needed. Saves time + lets BASE re-engage faster.
        self._stuck_trigger_dist: Optional[float] = None
        self._slope_trigger_dist: Optional[float] = None
        # Minimum net progress to consider an obstacle "passed". Without
        # this any tiny oscillation could fool the abort logic.
        self._post_commit_pass_margin_m = 0.5
        # Low-pass filter the published cmd to suppress left/right swaying
        # when the reactive sub-goal flips between adjacent sweep candidates.
        self._cmd_prev = np.zeros(2, dtype=np.float32)
        self._cmd_smooth_alpha = 0.6   # new = α*new + (1-α)*prev
        # Smooth the *high-level* sub-goal angle too. The actor is
        # stochastic (deterministic=False) so its theta_sub samples jitter
        # ±0.3-0.5 rad frame-to-frame — that translates to ω flips at the
        # low level. EMA at α=0.3 (heavy smoothing) lets the sub-goal
        # change gradually; r_sub is left untouched.
        self._theta_sub_prev: float = 0.0
        self._theta_sub_alpha: float = 0.3
        # ── Human teleop mode ────────────────────────────────────────────
        self._human_keys = set()
        self._human_listener = None
        if HUMAN_DRIVE:
            try:
                from pynput import keyboard as _kb
                def _on_press(key):
                    try:
                        k = key.char.lower() if hasattr(key, "char") and key.char else key.name
                    except Exception:
                        return
                    self._human_keys.add(k)
                def _on_release(key):
                    try:
                        k = key.char.lower() if hasattr(key, "char") and key.char else key.name
                    except Exception:
                        return
                    self._human_keys.discard(k)
                self._human_listener = _kb.Listener(
                    on_press=_on_press, on_release=_on_release)
                self._human_listener.daemon = True
                self._human_listener.start()
                self.get_logger().info(
                    "HUMAN_DRIVE=1 — WASD via pynput; demos saved to demos/")
            except Exception as e:
                self.get_logger().warn(f"HUMAN_DRIVE keyboard init failed: {e}")
        self._human_demo_steps = 0
        self._human_demo_path_idx = 0
        self._human_demo_buffer = []   # list of dicts per step
        # ── Interrupt system ──────────────────────────────────────────────
        # Base behavior = "go to target". Everything else (avoid pit, dodge
        # high slope, stuck recovery) is a *short* interrupt that holds the
        # cmd for a fixed number of steps, then yields back to base. This
        # keeps the rover always biased toward the goal between events.
        self._interrupt_steps = 0
        self._interrupt_cmd = np.zeros(2, dtype=np.float32)
        self._interrupt_label = "NONE"
        # Cooldown between interrupts of the same kind so a single obstacle
        # doesn't re-fire continuously. Counts down each step.
        self._interrupt_cooldown = {}
        # Post-slope commit: after ON_SLOPE rotation finishes, drive
        # forward at the new heading for this many steps before BASE
        # resumes goal-seeking. Otherwise BASE immediately turns back
        # toward the goal and pushes the rover into the same slope.
        self._post_slope_commit_steps = 0
        self._post_slope_commit_duration = 30
        # Tracking for RETRACE_RETURN: count avoidance hits during RETURN
        # phase. Once enough fire close together, switch to waypoint-by-
        # waypoint retrace of the outbound trajectory.
        self._return_avoid_hits = 0
        self._return_avoid_window_steps = 200    # ~20 s at 10 Hz
        self._return_avoid_trigger = 3
        self._retrace_mode = False
        # Episodic flag for safety interrupts (high tilt / blind cliff).
        self._high_tilt_thresh = 0.45            # ~26° roll/pitch combined — emergency back-up
        # Anti-rollover: fires only when tilt is genuinely dangerous.
        # Empirically 0.35 rad (~20°) was too aggressive — the rover
        # legitimately crosses 20° on most extreme-terrain climbs and
        # was being reversed away from every approach. Raised to 0.40 rad
        # (~23°) — between normal climb angles and the HIGH_TILT (26°).
        self._rollover_tilt_thresh = 0.48        # ~27° — was 0.40, too aggressive (suppressed legitimate climbs)
        # Combo trigger removed (caused false positives at any uphill
        # transition where rate just exceeds threshold).
        self._tilt_prev = 0.0
        self._rollover_active_steps = 0          # min reverse duration counter
        # Mid-tilt threshold: rover currently sitting on a meaningful slope
        # (~14°). BEV in body frame can't see this — it looks flat ahead
        # because the LiDAR is tilted along with the chassis — so we trust
        # IMU to detect "I'm on a slope right now, turn away".
        self._on_slope_thresh = 0.32        # ~18° — was 0.25, fired too often on minor terrain
        # Direction lock for the current slope event. Cleared only after
        # sustained flat ground (counter below), so a brief dip in tilt
        # between bumps doesn't flip the chosen side.
        self._slope_turn_dir = 0.0
        self._slope_flat_counter = 0
        self._slope_flat_required = 20   # 2 s at 10 Hz before resetting dir
        # Last slope direction chosen — used to alternate on tie-break
        # across separate slope events.
        self._last_slope_dir = 0.0
        # Sustained-tilt counter for ON_SLOPE trigger debouncing — only
        # fire when tilt stays above threshold for ≥3 control steps,
        # avoiding spikes from bumps / IMU noise.
        self._on_slope_tilt_high_steps = 0

        self.use_base_policy = USE_BASE_POLICY
        self.base_policy = FlagSeekingPolicy(FlagSeekingPolicyConfig())
        self.base_policy_mode = BASE_POLICY_MODE
        if self.use_base_policy:
            self.get_logger().info(
                f"Base policy ENABLED (mode={self.base_policy_mode}) — "
                f"flag-seeking demos fed via BC loss (set USE_BASE_POLICY=0 "
                f"to disable).")
        else:
            self.get_logger().info("Base policy DISABLED — learned actor only.")

        # ── Latest message buffers (overwritten by callbacks) ──────────────
        self._latest_points: Optional[np.ndarray] = None
        self._latest_imu: Optional[Imu] = None
        self._latest_odom: Optional[Odometry] = None
        self._msg_lock = threading.Lock()

        # ── Ground-truth pose poller (world frame, from gz) ────────────────
        # Used everywhere we need the rover's actual position so the policy's
        # goal vector lines up with the visual flag (which is also in world
        # frame). Odometry is kept only for body-frame velocity.
        self._latest_gt: Optional[Tuple[float, float, float]] = None
        self._gt_lock = threading.Lock()
        self._gt_thread_run = True
        self._gt_thread = threading.Thread(
            target=self._gt_poller, name="gt_poller", daemon=True)
        self._gt_thread.start()

        # ── World-frame BEV memory map ─────────────────────────────────────
        # Accumulates ground-occupancy readings across timesteps so the
        # base policy's reactive avoidance can distinguish "actually empty"
        # from "negative obstacle (pit/cliff)" — anywhere we previously
        # saw LiDAR returns stays marked as traversable even when the
        # current instantaneous BEV looks dark.
        self._mem_extent_m = 100.0          # world covers ±50 m
        self._mem_res_m = 0.25              # cell size
        self._mem_size = int(self._mem_extent_m / self._mem_res_m)   # 400
        self._memory_map = np.zeros(
            (self._mem_size, self._mem_size), dtype=np.float32)

        # ── Flag cleanup worker ────────────────────────────────────────────
        # gz `/world/<>/remove` calls can take >1 s under load and the
        # control loop runs at 10 Hz, so we hand off removals to a background
        # worker that retries on failure. Also sweep stale flags left over
        # from previous runs.
        self._pending_removals: "queue.Queue[Tuple[str, int]]" = queue.Queue()
        self._cleanup_thread_run = True
        self._cleanup_thread = threading.Thread(
            target=self._flag_cleanup_loop, name="flag_cleanup", daemon=True)
        self._cleanup_thread.start()
        self._sweep_stale_flags()

        # ── Perimeter walls ───────────────────────────────────────────────
        # Physically prevent the rover from rolling off the DEM heightmap.
        self._ensure_boundary_walls()

        # ── Episode bookkeeping ────────────────────────────────────────────
        self._episode_active = False
        self.episode_steps = 0
        self.episode_count = 0
        self.last_obs = None
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.last_distance_to_target: Optional[float] = None
        self.total_env_steps = 0
        self.total_train_steps = 0
        self._last_phase: Optional[MissionPhase] = None
        self._active_flag_name: Optional[str] = None

        # ── ROS subs / pubs ────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable_qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE,
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
        # Publish ground-truth rover pose so the viewer (or any other
        # external tool) can plot the *true* travel path, not odom.
        self.gt_pub = self.create_publisher(PoseStamped, "/td/rover_gt", 10)

        # ── Timers + training worker ───────────────────────────────────────
        self._dt = 1.0 / CONTROL_HZ
        self.create_timer(self._dt, self._control_step)
        self.create_timer(0.1, self._publish_gt)        # 10 Hz GT publisher
        self.create_timer(LOG_INTERVAL_S, self._log)
        self._last_checkpoint = time.time()

        self._train_stop = threading.Event()
        self._metrics_lock = threading.Lock()
        self._latest_metrics = {}
        # In HUMAN_DRIVE mode, skip the trainer thread entirely — we just
        # want the cmd to come from keyboard and demos to be saved.
        if not HUMAN_DRIVE:
            self._train_thread = threading.Thread(
                target=self._train_loop, name="td-train", daemon=True)
            self._train_thread.start()

        self.get_logger().info(
            "CraterTrainNode ready; waiting for sensors.")

    # ──────────────────────────────────────────────────────────────────────
    # Subscriber callbacks
    # ──────────────────────────────────────────────────────────────────────
    def _on_points(self, msg: PointCloud2) -> None:
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

    def _snapshot(self):
        with self._msg_lock:
            return (self._latest_points, self._latest_imu, self._latest_odom)

    # ──────────────────────────────────────────────────────────────────────
    # Ground-truth rover pose (world frame, from gz dynamic_pose/info)
    # ──────────────────────────────────────────────────────────────────────
    # Captures: x, y, qz, qw — enough to recover yaw. The dynamic_pose stream
    # interleaves every model in the world, so we anchor on the rover name
    # block. Gazebo's text output orders position then orientation per entity.
    _GZ_POSE_RE = re.compile(
        r'name:\s*"j100_0001/robot".*?position\s*\{\s*'
        r'x:\s*([-0-9.eE+]+)\s*y:\s*([-0-9.eE+]+)\s*z:\s*[-0-9.eE+]+\s*\}\s*'
        r'orientation\s*\{[^}]*?z:\s*([-0-9.eE+]+)\s*w:\s*([-0-9.eE+]+)',
        re.DOTALL,
    )

    def _update_and_overlay_memory(self, bev, px, py, yaw):
        """Fuse current BEV occupancy into the world-frame memory map, then
        return a 1-channel ego projection of the accumulated memory.

        The instantaneous BEV is **not** modified — only the SafetyShield
        and RSSM see the raw current frame; the BasePolicy uses the
        returned memory channel separately for negative-obstacle reasoning.
        """
        H, W = bev.shape[1], bev.shape[2]            # ego BEV grid size
        ego_extent_m = 15.0
        ego_cell = (2 * ego_extent_m) / H
        mem_res = self._mem_res_m
        mem_n = self._mem_size
        mem_half = self._mem_extent_m / 2.0

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # 1) Update world memory from current BEV occupancy.
        occ = bev[0]                                  # [H, W] in [0, 1]
        cur_ys, cur_xs = np.nonzero(occ > 0.3)
        if cur_ys.size:
            # BEV: row 0 = forward edge (max +x); col 0 = right (-y)
            #   x_ego =  ego_extent_m - row*ego_cell
            #   y_ego = -ego_extent_m + col*ego_cell
            x_ego = ego_extent_m - cur_ys * ego_cell
            y_ego = -ego_extent_m + cur_xs * ego_cell
            # Rotate to world frame (ego frame at rover pose).
            x_w = px + cos_y * x_ego - sin_y * y_ego
            y_w = py + sin_y * x_ego + cos_y * y_ego
            mem_col = ((x_w + mem_half) / mem_res).astype(np.int32)
            mem_row = ((y_w + mem_half) / mem_res).astype(np.int32)
            valid = (mem_col >= 0) & (mem_col < mem_n) \
                  & (mem_row >= 0) & (mem_row < mem_n)
            self._memory_map[mem_row[valid], mem_col[valid]] = 1.0

        # 2) Project memory back into ego BEV (overlay channel 0).
        # Build a meshgrid of ego cell centers and query memory.
        rows = np.arange(H)
        cols = np.arange(W)
        cc, rr = np.meshgrid(cols, rows)              # both [H, W]
        x_ego = ego_extent_m - rr * ego_cell
        y_ego = -ego_extent_m + cc * ego_cell
        x_w = px + cos_y * x_ego - sin_y * y_ego
        y_w = py + sin_y * x_ego + cos_y * y_ego
        mc = ((x_w + mem_half) / mem_res).astype(np.int32)
        mr = ((y_w + mem_half) / mem_res).astype(np.int32)
        valid = (mc >= 0) & (mc < mem_n) & (mr >= 0) & (mr < mem_n)
        memory_proj = np.zeros_like(occ)
        memory_proj[valid] = self._memory_map[mr[valid], mc[valid]]
        return memory_proj

    def _query_rover_gt(self) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, yaw) in world frame, or None on failure."""
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
            x = float(m.group(1)); y = float(m.group(2))
            qz = float(m.group(3)); qw = float(m.group(4))
            # Yaw from (qz, qw) — pitch/roll ignored (rover stays flat enough).
            yaw = 2.0 * math.atan2(qz, qw)
            return x, y, yaw
        except ValueError:
            return None

    def _gt_poller(self) -> None:
        """Background thread that refreshes ``self._latest_gt`` continuously.

        gz CLI calls take ~80–200 ms so this can't run inside the 10 Hz
        control loop. Holding the latest cached value is good enough because
        the rover moves slowly (≤0.6 m/s) and the policy is itself running
        at 10 Hz.
        """
        while self._gt_thread_run:
            gt = self._query_rover_gt()
            if gt is not None:
                with self._gt_lock:
                    self._latest_gt = gt
            time.sleep(0.05)

    # ──────────────────────────────────────────────────────────────────────
    # Gazebo flag spawn / remove
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
                 "--timeout", "4000",
                 "--req", f'name: "{name}" type: MODEL'],
                capture_output=True, text=True, timeout=8)
            return "data: true" in r_.stdout
        except Exception:
            return False

    # Background cleanup thread: retry on failure so the control loop never blocks.
    _MAX_REMOVE_RETRIES = 6
    _SWEEP_INTERVAL_S = 60.0     # how often to scan the world for stale flags

    def _flag_cleanup_loop(self) -> None:
        last_sweep = time.time()
        while self._cleanup_thread_run:
            # Periodic sweep — catches flags that escaped the per-episode
            # cleanup (e.g. all 6 removal retries timed out under gz load,
            # or the trainer crashed mid-episode). Anything starting with
            # td_goal_ that isn't the current active flag is re-queued.
            if time.time() - last_sweep > self._SWEEP_INTERVAL_S:
                last_sweep = time.time()
                self._enqueue_stale_goal_flags()
            try:
                name, retries = self._pending_removals.get(timeout=2.0)
            except queue.Empty:
                continue
            if self._gz_remove_entity(name):
                continue
            if retries < self._MAX_REMOVE_RETRIES:
                time.sleep(0.5 + 0.3 * retries)
                self._pending_removals.put((name, retries + 1))
            else:
                self.get_logger().warn(
                    f"[flag] gave up removing '{name}' after "
                    f"{self._MAX_REMOVE_RETRIES} retries (will be re-queued "
                    f"on next sweep)")

    def _enqueue_stale_goal_flags(self) -> None:
        """Scan the world for td_goal_* models and queue all of them
        (except the currently active flag) for removal."""
        try:
            r_ = subprocess.run(
                ["gz", "model", "--list"],
                capture_output=True, text=True, timeout=5)
        except Exception:
            return
        active = self._active_flag_name
        queued = 0
        for line in r_.stdout.splitlines():
            s = line.strip().lstrip("-").strip()
            if s.startswith("td_goal_") and s != active:
                self._pending_removals.put((s, 0))
                queued += 1
        if queued:
            self.get_logger().info(
                f"[flag] sweep enqueued {queued} stale flag(s)")

    def _spawn_wall(self, name: str, x: float, y: float,
                    sx: float, sy: float) -> bool:
        sdf = WALL_SDF_TEMPLATE.format(
            name=name, sx=sx, sy=sy, sz=WALL_HEIGHT_M,
            hz=WALL_CENTER_Z_M)
        req = (
            f"sdf: '{sdf}' "
            f'name: "{name}" '
            f"allow_renaming: false "
            f"pose {{ position {{ x: {x} y: {y} z: 0 }} "
            f"orientation {{ w: 1 }} }}"
        )
        try:
            r_ = subprocess.run(
                ["gz", "service", "-s", f"/world/{GAZEBO_WORLD}/create",
                 "--reqtype", "gz.msgs.EntityFactory",
                 "--reptype", "gz.msgs.Boolean",
                 "--timeout", "4000", "--req", req],
                capture_output=True, text=True, timeout=8)
            return "data: true" in r_.stdout
        except Exception as exc:
            self.get_logger().warn(f"[wall] spawn '{name}' failed: {exc}")
            return False

    def _ensure_boundary_walls(self) -> None:
        """Build 4 perimeter walls around the DEM (idempotent + retries).

        gz CLI calls can take >1 s under load, and a stale wall with the same
        name will make ``create`` fail silently. Strategy: try to remove the
        old wall first, then spawn; if spawn fails, remove + sleep + retry.
        """
        offset = WALL_HALF_M + WALL_THICKNESS_M / 2.0  # outer edge of wall
        walls = [
            # North/South walls run along X axis (sx long, sy thin)
            ("td_wall_n",  0.0,  offset, WALL_LENGTH_M, WALL_THICKNESS_M),
            ("td_wall_s",  0.0, -offset, WALL_LENGTH_M, WALL_THICKNESS_M),
            # East/West walls run along Y axis (sx thin, sy long)
            ("td_wall_e",  offset, 0.0, WALL_THICKNESS_M, WALL_LENGTH_M),
            ("td_wall_w", -offset, 0.0, WALL_THICKNESS_M, WALL_LENGTH_M),
        ]
        for name, _, _, _, _ in walls:
            self._gz_remove_entity(name)
        time.sleep(1.0)
        built = 0
        for name, x, y, sx, sy in walls:
            for attempt in range(4):
                if self._spawn_wall(name, x, y, sx, sy):
                    built += 1
                    break
                # spawn failed — maybe stale entity still alive. Retry after
                # an explicit remove + back-off.
                self._gz_remove_entity(name)
                time.sleep(0.5 + 0.5 * attempt)
        self.get_logger().info(
            f"[wall] built {built}/4 perimeter walls "
            f"(h={WALL_HEIGHT_M:.1f} m, top z={WALL_BOTTOM_Z_M + WALL_HEIGHT_M:.1f} m, "
            f"span=±{WALL_HALF_M} m)")

    def _sweep_stale_flags(self) -> None:
        """Sweep the world at startup; enqueue every td_goal_* model for removal."""
        try:
            r_ = subprocess.run(
                ["gz", "model", "--list"],
                capture_output=True, text=True, timeout=5)
        except Exception:
            return
        n = 0
        for line in r_.stdout.splitlines():
            s = line.strip().lstrip("-").strip()
            if s.startswith("td_goal_"):
                self._pending_removals.put((s, 0))
                n += 1
        if n:
            self.get_logger().info(
                f"[flag] queued {n} stale flag(s) from previous runs for cleanup")

    def _remove_active_flag(self) -> None:
        if self._active_flag_name:
            self._pending_removals.put((self._active_flag_name, 0))
            self._active_flag_name = None

    # ──────────────────────────────────────────────────────────────────────
    # Episode reset
    # ──────────────────────────────────────────────────────────────────────
    def _teleport_to_origin(self) -> bool:
        """Use gz set_pose to drop the rover at (0, 0, RESPAWN_HEIGHT_M) with
        random yaw, so every episode begins at the middle of the DEM."""
        yaw = random.uniform(-math.pi, math.pi)
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        req = (
            f'name: "j100_0001/robot" '
            f'position {{ x: 0 y: 0 z: {RESPAWN_HEIGHT_M} }} '
            f'orientation {{ z: {qz} w: {qw} }}'
        )
        try:
            r = subprocess.run(
                ["gz", "service", "-s", f"/world/{GAZEBO_WORLD}/set_pose",
                 "--reqtype", "gz.msgs.Pose",
                 "--reptype", "gz.msgs.Boolean",
                 "--timeout", "2000", "--req", req],
                capture_output=True, text=True, timeout=5.0)
            return "data: true" in r.stdout
        except Exception as exc:
            self.get_logger().warn(f"[reset] set_pose failed: {exc}")
            return False

    def _is_obstacle_world_xy(self, gx: float, gy: float) -> bool:
        """Look up the (gx, gy) world-frame point in the obstacle mask. The
        heightmap covers ±DEM_FULL_M/2 in world frame; PNG row 0 is at
        world +Y, column 0 at world -X (gz heightmap convention)."""
        mask = self._obstacle_mask
        if mask is None:
            return False
        H, W = mask.shape
        half = DEM_FULL_M / 2.0
        # Map (gx, gy) ∈ [-half, +half] → pixel.
        px = int((gx + half) / DEM_FULL_M * W)
        py = int((half - gy) / DEM_FULL_M * H)
        if 0 <= px < W and 0 <= py < H:
            return bool(mask[py, px])
        return True  # outside map → treat as obstacle

    def _sample_destination_in_world(self):
        """Sample a destination uniformly across the entire DEM (±DEM_HALF_M
        on each axis). Rejects samples too close to the origin so the policy
        always has a meaningful drive to do, and (if an obstacle mask is
        loaded for this world) rejects samples that land on non-traversable
        terrain so the flag is always reachable.
        """
        for _ in range(200):
            gx = random.uniform(-DEM_HALF_M, DEM_HALF_M)
            gy = random.uniform(-DEM_HALF_M, DEM_HALF_M)
            if math.hypot(gx, gy) < DEST_MIN_DISTANCE_M:
                continue
            if self._is_obstacle_world_xy(gx, gy):
                continue
            return (gx, gy)
        # Highly unlikely fallback: nothing traversable found in 200 tries.
        return (DEM_HALF_M * 0.5, 0.0)

    def _reset_episode(self, _unused=None):
        """Reset for a new episode — everything in world frame.

        We now use the rover's ground-truth pose (from the gz GT poller) for
        the policy's position input, so the mission FSM, the goal vector, and
        the visual flag all live in the same world frame. ``origin = (0, 0)``
        because the rover is teleported there at the start of every episode.
        """
        self._remove_active_flag()

        # 1. Physical reset — teleport rover to the centre of the heightmap.
        if not self._teleport_to_origin():
            self.get_logger().warn(
                "[reset] could not teleport rover to origin; continuing.")
        time.sleep(RESET_SETTLE_S)

        # 2. Sample a destination in world frame, inside the DEM.
        gx, gy = self._sample_destination_in_world()

        # 3. Re-init mission FSM in world frame.
        self.mission.reset(origin=(0.0, 0.0), destination=(gx, gy))
        self.trajectory.reset()
        self.episode_steps = 0
        self.episode_count += 1
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.last_obs = None
        self.last_distance_to_target = None
        self.rssm_state = None
        self._last_phase = MissionPhase.OUTBOUND
        self._episode_active = True

        # 4. Spawn the yellow flag at exactly the mission destination — same
        #    coordinate the policy sees in goal_vector.
        flag_name = f"td_goal_out_{self.episode_count}"
        if self._gz_spawn_flag(flag_name, gx, gy, YELLOW_RGB):
            self._active_flag_name = flag_name
            self.get_logger().info(
                f"[flag] yellow {flag_name} @ world=({gx:.2f}, {gy:.2f})")
        self.get_logger().info(
            f"=== Episode {self.episode_count}: world origin=(0.00, 0.00) "
            f"dest=({gx:.2f}, {gy:.2f}) ===")

    def _on_phase_transition(self, new_phase: MissionPhase) -> None:
        if new_phase == MissionPhase.RETURN:
            self._remove_active_flag()
            # Rover was teleported to world (0, 0) at episode reset, so the
            # mission's odom-frame origin maps back to world (0, 0). Drop the
            # return flag there directly — no ground-truth query needed.
            flag_name = f"td_goal_home_{self.episode_count}"
            if self._gz_spawn_flag(flag_name, 0.0, 0.0, RED_RGB):
                self._active_flag_name = flag_name
                self.get_logger().info(
                    f"[flag] yellow→red {flag_name} @ world=(0.00, 0.00) "
                    f"(rover reached destination, returning home)")

    # ──────────────────────────────────────────────────────────────────────
    # Observation construction
    # ──────────────────────────────────────────────────────────────────────
    def _build_observation(self, points, imu_msg, odom_msg):
        # Use ground-truth pose for position + yaw so the policy's goal vector
        # matches the visual flag (both world frame). Odometry still supplies
        # body-velocity / IMU only — those are first-derivative and not
        # affected by absolute-position drift.
        q = odom_msg.pose.pose.orientation
        roll, pitch, _odom_yaw = quat_to_rpy(q.x, q.y, q.z, q.w)

        with self._gt_lock:
            gt = self._latest_gt
        if gt is None:
            # GT poller hasn't published yet (just after startup or restart).
            # Fall back to odom so we don't drop the very first observations;
            # rover is at world (0, 0) post-teleport so error is tiny.
            px = odom_msg.pose.pose.position.x
            py = odom_msg.pose.pose.position.y
            yaw = _odom_yaw
        else:
            px, py, yaw = gt

        if not self._episode_active:
            self._reset_episode()

        self.mission.update((px, py, yaw), self.trajectory)
        if self.mission.phase != self._last_phase:
            self._on_phase_transition(self.mission.phase)
            self._last_phase = self.mission.phase

        # RETRACE_RETURN override: when the episode keeps hitting bad
        # terrain on the way home, switch the base target from "straight
        # at origin (0, 0)" to the recorded outbound waypoints, walked in
        # reverse. The interrupt detector still arbitrates per-step on
        # top of this; we're only changing where BASE points.
        if (self._retrace_mode and
                self.mission.phase.name == "RETURN"):
            while self.trajectory.advance_if_reached((px, py, yaw)):
                pass
            wp = self.trajectory.get_current_return_waypoint()
            if wp is not None:
                self.mission._current_target = (wp[0], wp[1])

        bev = points_to_bev(points, grid_size=self.cfg.model.bev_shape[1],
                            extent_m=15.0)
        # Build the visited-ground memory channel only when the memory
        # base-policy mode is active — otherwise it's unused and we save
        # the projection cost.
        memory_proj = None
        if self.base_policy_mode == "memory":
            try:
                memory_proj = self._update_and_overlay_memory(bev, px, py, yaw)
            except Exception:
                memory_proj = None
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
        tgt = self.mission.get_current_target()
        goal_vec = build_goal_vec((px, py), yaw, tgt)
        phase = self.mission.get_phase_vector()

        obs = {
            "lidar_bev":     torch.from_numpy(bev),
            "imu":           torch.from_numpy(imu_vec),
            **({"lidar_bev_memory": torch.from_numpy(memory_proj)}
                if memory_proj is not None else {}),
            "goal_vector":   torch.from_numpy(goal_vec),
            "mission_phase": torch.tensor(phase, dtype=torch.float32),
            "prev_action":   torch.from_numpy(self.prev_action.copy().astype(np.float32)),
        }

        # Bearing to target in body frame, [-pi, pi]; >0 means goal is on
        # the rover's left.
        _dx_t = tgt[0] - px
        _dy_t = tgt[1] - py
        _bearing = math.atan2(
            -math.sin(yaw) * _dx_t + math.cos(yaw) * _dy_t,
            math.cos(yaw) * _dx_t + math.sin(yaw) * _dy_t,
        )
        info = {
            "rover_xy": (px, py),
            "yaw": yaw,
            "tilt": math.hypot(roll, pitch),
            "phase": self.mission.phase,
            "distance_to_target": float(math.hypot(_dx_t, _dy_t)),
            "bearing_to_target": _bearing,
            "target": tgt,
        }
        return obs, info

    # ──────────────────────────────────────────────────────────────────────
    # Reward + termination
    # ──────────────────────────────────────────────────────────────────────
    def _reward_and_done(self, obs, info):
        rcfg = self.cfg.reward
        rew = rcfg.step_penalty
        if self.last_distance_to_target is not None:
            progress = self.last_distance_to_target - info["distance_to_target"]
            rew += rcfg.progress_weight * progress
        rew += rcfg.angular_penalty * abs(float(self.prev_action[1]))

        # ── Continuous obstacle-avoidance signals ──────────────────────────
        # 1) Tilt: quadratic penalty above threshold so the actor sees a
        #    gradient before reaching the hard fail at TILT_FAIL_RAD.
        tilt = info["tilt"]
        if tilt > rcfg.tilt_penalty_threshold_rad:
            d = tilt - rcfg.tilt_penalty_threshold_rad
            rew -= rcfg.tilt_penalty_weight * (d * d)
        # 2) Lidar proximity: occupancy channel (channel 0) of the BEV.
        #    Inspect a small central window so we only flag obstacles within
        #    ~3.5 m of the rover. Penalty is linear above the threshold.
        try:
            occ = obs["lidar_bev"][0]   # tensor or ndarray [H, W]
            H = occ.shape[-1]
            r = max(1, H // 8)          # ~ 3.5 m at BEV extent 15 m
            c = H // 2
            if hasattr(occ, "numpy"):
                near = float(occ[c - r:c + r, c - r:c + r].max().item())
            else:
                near = float(occ[c - r:c + r, c - r:c + r].max())
            if near > rcfg.lidar_penalty_threshold:
                rew -= rcfg.lidar_penalty_weight * (near - rcfg.lidar_penalty_threshold)
        except Exception:
            pass

        done = False
        if tilt > TILT_FAIL_RAD:
            rew += rcfg.collision_penalty
            done = True
        if self.mission.is_success():
            rew += rcfg.success_bonus
            done = True
        if self.episode_steps >= int(EPISODE_TIMEOUT_S * CONTROL_HZ):
            rew += rcfg.timeout_penalty
            done = True
        return float(rew), bool(done)

    # ──────────────────────────────────────────────────────────────────────
    # Publish cmd_vel
    # ──────────────────────────────────────────────────────────────────────
    def _update_stuck_state(self, px, py, now, tilt=0.0, dist_to_target=None,
                            ang_vel_z=None):
        """Track GT positions + goal distance + IMU rotation; set
        self._stuck_count > 0 when the rover is geometrically stuck AND
        IMU confirms (low actual yaw rate). Combining both sensors
        reduces false positives — e.g. rotating in place looks stuck by
        position but yaw rate confirms motion is happening.

        Two independent triggers (either one increments the counter):
          (a) bbox of recent positions stays within self._stuck_min_dist_m
              AND IMU yaw rate is low (no useful rotation either)
          (b) min distance-to-target over the window minus the latest
              distance < self._stuck_min_progress_m
              (not getting closer to the goal)

        Tilt-aware (×3 increment when tilt > 0.4 rad). When the counter
        drops back to 0, fire a post-stuck commit period.
        """
        # While a scripted recovery is in flight, freeze stuck detection
        # entirely. Rotation during recovery would otherwise keep
        # progress_stuck true and re-trigger forever.
        if self._recovery_phase != 0:
            self._stuck_count = 0
            self._stuck_history = []
            return
        prev_count = self._stuck_count
        d = float(dist_to_target) if dist_to_target is not None else float("nan")
        self._stuck_history.append((now, px, py, d))
        cutoff = now - self._stuck_window_s
        self._stuck_history = [h for h in self._stuck_history if h[0] >= cutoff]
        if len(self._stuck_history) < 10:
            self._stuck_count = 0
            return
        xs = [h[1] for h in self._stuck_history]
        ys = [h[2] for h in self._stuck_history]
        bbox = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        # IMU confirmation: bbox-stuck only counts when the rover ALSO
        # isn't producing meaningful yaw rotation. If |ω_z| > 0.3 rad/s
        # then the rover is at least turning in place — wait before
        # marking it as stuck.
        low_rotation = (ang_vel_z is None) or (abs(ang_vel_z) < 0.3)
        bbox_stuck = (bbox < self._stuck_min_dist_m) and low_rotation

        # Goal-progress check: the closest the rover has been to the
        # target inside the window vs the current distance. If we haven't
        # advanced by at least min_progress, we're either oscillating or
        # being pushed back by slope slip.
        ds = [h[3] for h in self._stuck_history if not math.isnan(h[3])]
        progress_stuck = False
        if len(ds) >= 5 and not math.isnan(d):
            min_d = min(ds[:-1])      # exclude current sample
            progress_stuck = (min_d - d) < self._stuck_min_progress_m

        # Require BOTH bbox stuck AND no progress before counting as
        # stuck. Previously OR fired false positives on flat terrain
        # where the rover briefly slowed (bbox small) but was still
        # progressing → recovery reverse caused visible "flashback".
        # Extra rescue: if rover gained >0.5 m in the last 2 s, the
        # situation is fine; force-reset the counter even if bbox is
        # still small.
        if bbox_stuck and progress_stuck:
            self._stuck_count += 3 if tilt > 0.4 else 1
        else:
            # Detect strong recent progress (within last 2 s).
            strong_recent_progress = False
            if len(ds) >= 4:
                strong_recent_progress = (ds[-4] - ds[-1]) > 0.5
            if prev_count > 30 and not strong_recent_progress:
                self._post_stuck_commit_steps = self._post_stuck_commit_duration
            self._stuck_count = 0
            self._stuck_rotation_target_yaw = None
            self._stuck_recovery_dir = 0.0

    def _strong_recent_progress(self, window_s: float = 2.0,
                                  threshold_m: float = 0.5) -> bool:
        """Travel-path signal: did the rover gain >threshold_m on its
        distance-to-target within the last window_s seconds?
        Returns True when the rover is healthily progressing — used to
        gate avoidance triggers and prevent false positives.
        """
        if len(self._stuck_history) < 4:
            return False
        n_frames = max(2, int(window_s * CONTROL_HZ))
        ds = [h[3] for h in self._stuck_history if not math.isnan(h[3])]
        if len(ds) < n_frames + 1:
            return False
        return (ds[-n_frames - 1] - ds[-1]) > threshold_m

    def _imu_concur_for_avoidance(self, tilt_now: float,
                                     ang_vel_z: Optional[float]) -> bool:
        """IMU signal: rover is showing motion patterns consistent with
        approaching trouble — tilt above 0.10 rad OR strong yaw rate.
        Returns True when IMU concurs that something non-flat is happening.
        """
        if tilt_now > 0.10:
            return True
        if ang_vel_z is not None and abs(ang_vel_z) > 0.5:
            return True
        return False

    def _avoidance_should_fire(self, lidar_evidence: bool,
                                  tilt_now: float,
                                  ang_vel_z: Optional[float],
                                  *, require_lidar: bool = True) -> bool:
        """Comprehensive 3-signal gate for BEV-based avoidance triggers
        (HIGH_SLOPE, NEG_OBSTACLE, ON_SLOPE secondary). Combines:
          1. LiDAR (BEV) — direct evidence of obstacle/slope/pit
          2. Travel path — refuse to fire if rover is progressing fine
          3. IMU — refuse to fire if everything is flat (LiDAR noise)
        Fire when: LiDAR + at least one of {IMU concurs, path stalled}.
        """
        if require_lidar and not lidar_evidence:
            return False
        if self._strong_recent_progress():
            return False    # rover is fine, let it through
        imu_ok = self._imu_concur_for_avoidance(tilt_now, ang_vel_z)
        return imu_ok or not self._strong_recent_progress()

    def _on_slope_combined_trigger(self, obs, tilt_now):
        """Combined IMU + BEV + path decision for ON_SLOPE. Trigger when:
          (a) IMU tilt sustained above 0.32 rad for ≥3 frames (≥0.3s)
              AND path shows no strong recent progress; OR
          (b) IMU tilt > 0.18 rad AND BEV shows ≥2 bad rays one side
              AND path shows no strong recent progress.
        The path gate prevents ON_SLOPE from firing while the rover is
        successfully climbing — if dist_to_target is dropping >0.5 m/2s,
        the slope is being traversed fine, don't interfere.
        Returns bool.
        """
        # Path gate: if rover is healthily progressing, NEVER fire
        # ON_SLOPE — it's climbing the slope successfully.
        if self._strong_recent_progress():
            self._on_slope_tilt_high_steps = 0
            return False
        # Track sustained-tilt counter on the instance.
        if tilt_now > self._on_slope_thresh:
            self._on_slope_tilt_high_steps = getattr(
                self, "_on_slope_tilt_high_steps", 0) + 1
        else:
            self._on_slope_tilt_high_steps = 0
        if self._on_slope_tilt_high_steps >= 3:
            return True
        if tilt_now > 0.18:
            try:
                bad_l, bad_r, _, _ = self._bev_side_scores(obs)
                if max(bad_l, bad_r) >= 2:
                    return True
            except Exception:
                pass
        return False

    def _bev_side_scores(self, obs):
        """Score forward fan into left vs right "bad terrain" counts.
        Returns (score_left, score_right) — higher = worse on that side.
        Used by both HIGH_SLOPE and ON_SLOPE so the turn direction comes
        from real BEV slope/pit distribution, not goal-bearing heuristics.
        Returns (0, 0) when no LiDAR signal — caller falls back to its
        own heuristic.
        """
        bev = obs.get("lidar_bev")
        if bev is None:
            return 0.0, 0.0
        occ = bev[0]
        elev = bev[2]
        rough = bev[3]    # roughness channel — used to bias toward smoother side
        H, W = occ.shape
        ch, cw = H // 2, W // 2
        extent_m = 15.0
        elev_span_m = 4.5
        wedge_deg = 30.0
        look_m = 6.0
        N_rays = 7
        N_samples = 10
        # Ego ground.
        pad = 3
        patch_occ = occ[max(0, ch - pad):ch + pad + 1,
                        max(0, cw - pad):cw + pad + 1]
        patch_elev = elev[max(0, ch - pad):ch + pad + 1,
                          max(0, cw - pad):cw + pad + 1]
        m = patch_occ > 0.5
        ego_elev = float(patch_elev[m].mean()) if m.any() else 0.5
        px_per_m = H / extent_m
        ray_len_px = min(look_m * px_per_m, min(H, W) / 2.0 - 2)
        ts = torch.linspace(0.4, 1.0, N_samples)
        thetas = torch.linspace(
            math.radians(-wedge_deg / 2.0),
            math.radians(wedge_deg / 2.0),
            N_rays,
        )
        neg_drop = 0.08
        slope_tan = math.tan(math.radians(20.0))
        step_run_m = look_m * (ts[1] - ts[0]).item()
        # Per-side accumulators: bad-ray count + mean elevation + mean
        # roughness. Final score = bad_count + α·(elev_rise) + β·rough.
        # Direction chosen by caller toward LOWER score (smoother, lower).
        s_left = s_right = 0.0
        elev_left_sum = elev_right_sum = 0.0
        rough_left_sum = rough_right_sum = 0.0
        n_left = n_right = 0
        for i, theta in enumerate(thetas):
            theta = float(theta)
            dx = (ray_len_px * ts) * math.cos(theta)
            dy = (ray_len_px * ts) * math.sin(theta)
            rows = (ch - dx).long().clamp(0, H - 1)
            cols = (cw + dy).long().clamp(0, W - 1)
            ray_occ = occ[rows, cols]
            ray_elev = elev[rows, cols]
            ray_rough = rough[rows, cols]
            drop = ego_elev - ray_elev
            pit_frac = float(((ray_occ > 0.5) & (drop > neg_drop)).float().mean())
            diffs = (ray_elev[1:] - ray_elev[:-1]).clamp_min(0.0) * elev_span_m
            slope_frac = float(
                (((diffs / max(step_run_m, 1e-3)) > slope_tan)
                 & (ray_occ[1:] > 0.5) & (ray_occ[:-1] > 0.5)).float().mean())
            bad = pit_frac > 0.15 or slope_frac > 0.18
            # Mean ray elevation rise vs ego (positive = rising terrain).
            obs_mask = ray_occ > 0.5
            if obs_mask.any():
                mean_elev_rise = float(
                    (ray_elev[obs_mask] - ego_elev).mean())
                mean_rough = float(ray_rough[obs_mask].mean())
            else:
                mean_elev_rise = 0.0
                mean_rough = 0.0
            if i < N_rays / 2.0:
                if bad:
                    s_right += 1
                elev_right_sum += mean_elev_rise   # left ray data → right side score
                rough_right_sum += mean_rough
                n_right += 1
            if i > (N_rays - 1) / 2.0:
                if bad:
                    s_left += 1
                elev_left_sum += mean_elev_rise
                rough_left_sum += mean_rough
                n_left += 1
        # Direction preference score (continuous): bad-count + elev rise
        # + roughness. Used only to PICK the smoother / lower side once
        # a trigger has already fired. NEVER mix into the trigger
        # threshold — its float drift causes false positives on flat
        # terrain with mild elevation gradient or LiDAR noise.
        dir_left = s_left
        dir_right = s_right
        if n_left > 0:
            dir_left += 4.0 * (elev_left_sum / n_left)
            dir_left += 2.0 * (rough_left_sum / n_left)
        if n_right > 0:
            dir_right += 4.0 * (elev_right_sum / n_right)
            dir_right += 2.0 * (rough_right_sum / n_right)
        # Return (trigger_left, trigger_right, dir_left, dir_right).
        # First pair is integer bad-ray counts for THRESHOLD use.
        # Second pair is continuous score for DIRECTION choice.
        return s_left, s_right, dir_left, dir_right

    def _detect_terrain_interrupt(self, obs):
        """BEV-based interrupt detector. Looks straight ahead in the
        rover's body frame and decides whether an avoidance maneuver
        is needed. Returns a dict {cmd, duration, label} or None.

        Priority within terrain detectors:
          NEG_OBSTACLE (pit) — strongest, dodge sideways
          HIGH_SLOPE   — turn around to a more traversable side
        Each event fires for a fixed step duration so the rover commits
        to the maneuver rather than flickering. A per-label cooldown
        prevents the same event from re-firing immediately.
        """
        bev = obs.get("lidar_bev")
        if bev is None:
            return None
        # obs tensor shape: [4, H, W]; rover at center, +X = forward.
        occ = bev[0]
        elev = bev[2]
        H, W = occ.shape
        ch, cw = H // 2, W // 2
        extent_m = 15.0
        elev_span_m = 4.5

        # Forward fan: take a narrow strip directly in front of the rover.
        # Rows 0..ch are the forward half (the BEV builder maps forward to
        # decreasing row). Sample a 30° wedge × 8 m ahead.
        wedge_deg = 30.0
        look_m = 6.0
        N_rays = 7
        N_samples = 10

        # Local ground under rover.
        pad = 3
        patch_occ = occ[max(0, ch - pad):ch + pad + 1,
                        max(0, cw - pad):cw + pad + 1]
        patch_elev = elev[max(0, ch - pad):ch + pad + 1,
                          max(0, cw - pad):cw + pad + 1]
        mask = patch_occ > 0.5
        ego_elev = float(patch_elev[mask].mean()) if mask.any() else 0.5

        px_per_m = H / extent_m
        ray_len_px = min(look_m * px_per_m, min(H, W) / 2.0 - 2)
        ts = torch.linspace(0.4, 1.0, N_samples)
        # Sample N_rays angles uniformly across the forward wedge.
        thetas = torch.linspace(
            math.radians(-wedge_deg / 2.0),
            math.radians(wedge_deg / 2.0),
            N_rays,
        )

        neg_drop = 0.08
        # Detect at 20° because the rover wheels start to slip well below
        # 30° on lunar-regolith friction — by the time the slope reads
        # 30° in BEV the rover is already on it, climbing and sliding back.
        slope_tan = math.tan(math.radians(20.0))
        step_run_m = look_m * (ts[1] - ts[0]).item()

        # LOST_VISION: forward fan returns essentially no LiDAR hits.
        # This usually means a cliff edge — beyond the lip the laser
        # sees only sky/empty. Slamming the brakes is safer than
        # creeping forward into the void.
        fwd_strip = occ[max(0, ch - 24):ch, max(0, cw - 12):cw + 12]
        fwd_fill = float((fwd_strip > 0).float().mean())
        if (fwd_fill < 0.05
                and "LOST_VISION" not in self._interrupt_cooldown):
            self._interrupt_cooldown["LOST_VISION"] = 30
            return {"cmd": np.array([-0.2, 0.0], dtype=np.float32),
                    "duration": 10, "label": "LOST_VISION"}

        any_pit = False
        any_steep = False
        # Compare which side (left/right) of the wedge has worse terrain
        # so we know which way to dodge.
        score_left = 0.0
        score_right = 0.0

        for i, theta in enumerate(thetas):
            theta = float(theta)
            dx = (ray_len_px * ts) * math.cos(theta)
            dy = (ray_len_px * ts) * math.sin(theta)
            rows = (ch - dx).long().clamp(0, H - 1)
            cols = (cw + dy).long().clamp(0, W - 1)
            ray_occ = occ[rows, cols]
            ray_elev = elev[rows, cols]
            drop = ego_elev - ray_elev
            pit_cells = (ray_occ > 0.5) & (drop > neg_drop)
            pit_frac = float(pit_cells.float().mean())
            diffs = (ray_elev[1:] - ray_elev[:-1]).clamp_min(0.0) * elev_span_m
            slope_cells = (diffs / max(step_run_m, 1e-3)) > slope_tan
            obs_pair = (ray_occ[1:] > 0.5) & (ray_occ[:-1] > 0.5)
            slope_frac = float((slope_cells & obs_pair).float().mean())
            ray_bad = pit_frac > 0.15 or slope_frac > 0.18
            # Count bad rays by side. Mid ray (i == N_rays//2) does count
            # for the "force the rover to deviate" decision but is split
            # evenly between sides for direction selection.
            if ray_bad:
                if i < N_rays / 2.0:
                    score_right += 1   # left ray bad → prefer turning right
                if i > (N_rays - 1) / 2.0:
                    score_left += 1
            if pit_frac > 0.15:
                any_pit = True
            if slope_frac > 0.18:
                any_steep = True

        bad_ray_count = score_left + score_right
        # Block forward motion as soon as 3+ rays are bad (out of 7).
        forward_blocked = bad_ray_count >= 3

        # Path gate: don't fire any avoidance if rover is making clear
        # progress toward the goal — terrain is being traversed fine.
        if self._strong_recent_progress():
            return None

        # NEG_OBSTACLE: pits are dangerous → fire regardless of IMU.
        if any_pit and "NEG_OBSTACLE" not in self._interrupt_cooldown:
            self._interrupt_cooldown["NEG_OBSTACLE"] = 10
            turn = 0.8 if score_left >= score_right else -0.8
            return {"cmd": np.array([0.0, turn], dtype=np.float32),
                    "duration": 10, "label": "NEG_OBSTACLE"}

        # HIGH_SLOPE: BEV slope ahead + path stalled. (IMU not required
        # because rover may still be at the base of the slope.)
        if any_steep and forward_blocked and \
                "HIGH_SLOPE" not in self._interrupt_cooldown:
            self._interrupt_cooldown["HIGH_SLOPE"] = 12
            turn = 0.8 if score_left >= score_right else -0.8
            return {"cmd": np.array([0.0, turn], dtype=np.float32),
                    "duration": 18, "label": "HIGH_SLOPE"}
        return None

    def _stuck_recovery_cmd(self):
        """Scripted recovery — runs as a fixed phase sequence; the caller
        only invokes us while ``_recovery_phase != 0``. Phase transitions
        are driven by ``_recovery_step``; once phase 4 (cooldown) ends,
        we set phase=0 and the rover returns to BASE.
        Direction is locked per recovery event (alternates with the
        previous one so the rover doesn't repeatedly try the same side).
        Returns [v, omega].
        """
        # Pick direction on entry to phase 1.
        if self._recovery_phase == 1 and self._stuck_recovery_dir == 0.0:
            self._stuck_recovery_dir = (
                -self._last_stuck_dir if getattr(
                    self, "_last_stuck_dir", 0.0) != 0.0 else 1.0)
            self._last_stuck_dir = self._stuck_recovery_dir

        sign = self._stuck_recovery_dir
        step = self._recovery_step
        phase = self._recovery_phase

        # Phase A — reverse 1 s, NO rotation (straight back via wheels
        # only; rotation is handled in Phase B). This avoids the rover
        # immediately changing heading while still wedged in the obstacle.
        if phase == 1:
            if step >= 10:
                self._recovery_phase = 2
                self._recovery_step = 0
                phase = 2
                step = 0
        if phase == 1:
            return np.array([-0.25, 0.0], dtype=np.float32)
        # Phase B — rotate ~90° in 2 s
        if phase == 2:
            if step >= 20:
                self._recovery_phase = 3
                self._recovery_step = 0
                phase = 3
                step = 0
        if phase == 2:
            return np.array([0.0, sign * 0.8], dtype=np.float32)
        # Phase C — drive forward 3.5 s in the new heading
        if phase == 3:
            if step >= 35:
                self._recovery_phase = 4
                self._recovery_step = 0
                phase = 4
                step = 0
        if phase == 3:
            return np.array([0.4, 0.0], dtype=np.float32)
        # Phase D — cooldown (no override, but stuck detection frozen).
        # We just publish BASE-equivalent zeros; the caller will fall
        # through to BASE actor cmd via priority arbitration.
        return None

    def _subgoal_to_velocity(self, theta_sub, r_sub, gpu_obs):
        """Hierarchical-mode: convert a (theta_sub, r_sub) sub-goal in the
        rover's body frame into a low-level [linear_x, angular_z] command
        by feeding a synthetic goal_vector into the FlagSeekingPolicy.

        Sign convention: r_sub > 0 means forward, r_sub < 0 means reverse.
        Magnitude is the look-ahead distance. This lets the hierarchical
        action space represent reverse maneuvers (used by stuck recovery)
        without losing fidelity in demo distillation.

        Returns a numpy array shape ``[2]``.
        """
        device = next(self.model.parameters()).device
        r_abs = abs(r_sub)
        dx = r_abs * math.cos(theta_sub)
        dy = r_abs * math.sin(theta_sub)
        synth_goal = torch.tensor(
            [[dx, dy, r_abs, theta_sub]], dtype=torch.float32, device=device)
        sub_obs = {k: v for k, v in gpu_obs.items()}
        sub_obs["goal_vector"] = synth_goal
        vel_t = self.base_policy.compute(sub_obs)
        vel = vel_t[0].detach().cpu().numpy()
        if r_sub < 0.0:
            # Invert linear velocity to encode reverse motion. ω is kept
            # so the rover can still turn while reversing if needed.
            vel = np.array([-vel[0], vel[1]], dtype=np.float32)
        return vel

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

    def _publish_gt(self):
        """Publish the latest cached ground-truth rover pose so external
        tools (the viewer) can plot the *true* travel path, not drifting
        odometry."""
        with self._gt_lock:
            gt = self._latest_gt
        if gt is None:
            return
        x, y, yaw = gt
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        try:
            self.gt_pub.publish(msg)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────
    # 10 Hz control loop
    # ──────────────────────────────────────────────────────────────────────
    def _control_step(self):
        points, imu_msg, odom_msg = self._snapshot()
        if points is None or imu_msg is None or odom_msg is None:
            return

        obs, info = self._build_observation(points, imu_msg, odom_msg)

        device = next(self.model.parameters()).device
        gpu_obs = {k: v.unsqueeze(0).to(device) for k, v in obs.items()}

        # ── Action selection ────────────────────────────────────────────
        # In hierarchical mode the actor's output is a sub-goal in rover-frame
        # polar coords (theta, r); the FlagSeekingPolicy translates that into
        # a low-level [linear_x, angular_z] command. The BC label is the
        # "ideal" sub-goal generated by BasePolicy.compute_subgoal.
        is_hierarchical = self.cfg.model.use_hierarchical_action
        if self.use_base_policy:
            if is_hierarchical:
                base_action_t = self.base_policy.compute_subgoal(
                    gpu_obs, mode=self.base_policy_mode)
            else:
                base_action_t = self.base_policy.compute(gpu_obs)
            # Deterministic=True at deployment so the published cmd is the
            # actor's *mean* (not a sample) — kills the ω sign flips that
            # come from per-frame stochastic sampling and look like sway.
            # Training rollouts still get exploration via env noise + the
            # actor entropy bonus during the imagined-rollout loss.
            action_t, self.rssm_state = self.model.select_action(
                gpu_obs, state=self.rssm_state,
                human_action=base_action_t, control_mode=CONTROL_MODE_HUMAN,
                deterministic=True,
            )
            control_mode_for_log = CONTROL_MODE_HUMAN
        else:
            action_t, self.rssm_state = self.model.select_action(
                gpu_obs, state=self.rssm_state,
                human_action=None, control_mode=CONTROL_MODE_AUTO,
                deterministic=True,
            )
            control_mode_for_log = CONTROL_MODE_AUTO
        action_np = action_t[0].detach().cpu().numpy()

        # Convert hierarchical action → low-level [v, omega] before publishing.
        if is_hierarchical:
            theta_sub_raw, r_sub = float(action_np[0]), float(action_np[1])
            # EMA smooth the sub-goal heading so frame-to-frame stochastic
            # actor noise can't flip the low-level ω back and forth.
            a = self._theta_sub_alpha
            theta_sub = a * theta_sub_raw + (1.0 - a) * self._theta_sub_prev
            self._theta_sub_prev = theta_sub
            cmd_action_np = self._subgoal_to_velocity(theta_sub, r_sub, gpu_obs)
        else:
            cmd_action_np = action_np
        # Snapshot of actor-derived cmd before any interrupt overrides
        # — used by the ABLATE_INTERRUPTS ablation to restore the
        # un-overridden command at the end of arbitration.
        _actor_cmd_snapshot = cmd_action_np.copy()

        # ── HUMAN_DRIVE override ──────────────────────────────────────────
        # When HUMAN_DRIVE=1, replace the actor cmd with WASD keyboard
        # input. All other infrastructure (flags, mission, interrupts) is
        # left intact — interrupts still fire as safety overrides if the
        # human commands something dangerous (e.g. high tilt).
        if HUMAN_DRIVE:
            keys = self._human_keys
            v = 0.0; w = 0.0
            if "w" in keys: v += 0.6
            if "s" in keys: v -= 0.4
            if "a" in keys: w += 0.6
            if "d" in keys: w -= 0.6
            if "space" in keys:
                v = 0.0; w = 0.0
            cmd_action_np = np.array([v, w], dtype=np.float32)

        # ── Stuck detection + recovery override ───────────────────────────
        # If GT pose hasn't moved enough over the past few seconds, override
        # the cmd with an in-place rotation to escape. This is the dominant
        # failure mode observed in rugged terrain (mid-failures).
        with self._gt_lock:
            gt = self._latest_gt
        if gt is not None:
            try:
                q = odom_msg.pose.pose.orientation
                roll_s, pitch_s, _ = quat_to_rpy(q.x, q.y, q.z, q.w)
                tilt_now = math.hypot(roll_s, pitch_s)
            except Exception:
                tilt_now = 0.0
            # Pass IMU yaw rate so stuck detection can verify the rover
            # isn't simply rotating in place (which appears stuck by bbox
            # but is actually deliberate motion).
            try:
                ang_z = float(imu_msg.angular_velocity.z)
            except Exception:
                ang_z = None
            self._update_stuck_state(
                gt[0], gt[1], time.time(), tilt=tilt_now,
                dist_to_target=info.get("distance_to_target"),
                ang_vel_z=ang_z)
            # Reset ON_SLOPE direction lock only after SUSTAINED flat
            # ground (≥2 s under 0.10 rad tilt). Brief dips in tilt while
            # bouncing across bumps must NOT clear the lock — otherwise
            # the next slope event picks a fresh (possibly opposite) side.
            if tilt_now < 0.10:
                self._slope_flat_counter += 1
                if (self._slope_flat_counter >= self._slope_flat_required
                        and self._slope_turn_dir != 0.0):
                    self._last_slope_dir = self._slope_turn_dir
                    self._slope_turn_dir = 0.0
            else:
                self._slope_flat_counter = 0
            # ── Interrupt arbitration (priority high → low) ───────────────
            # Base behavior cmd_action_np is "go to target" (already set
            # above). Each branch below briefly overrides for an event.
            # First match wins; everything else yields back to base.

            # P-1: ANTI_ROLLOVER — top priority. Fires only when tilt
            # exceeds 0.40 rad (~23°) — close to but below HIGH_TILT.
            # Action: pure reverse at -0.25 m/s, NO rotation. Holds for
            # at least 10 frames (1 s) so the wheels actually translate.
            self._tilt_prev = tilt_now   # kept for future use
            if tilt_now > self._rollover_tilt_thresh:
                self._rollover_active_steps = max(
                    self._rollover_active_steps, 10)
            if self._rollover_active_steps > 0:
                cmd_action_np = np.array([-0.25, 0.0], dtype=np.float32)
                self._rollover_active_steps -= 1
                self._interrupt_label = "ANTI_ROLLOVER"
            # P0: safety — milder high-tilt back-up (kept as fallback if
            # ANTI_ROLLOVER fires but tilt remains high).
            elif tilt_now > self._high_tilt_thresh:
                cmd_action_np = np.array([-0.15, 0.0], dtype=np.float32)
                self._interrupt_label = "HIGH_TILT"
                self._interrupt_steps = 0       # let condition re-check next frame
            # P0.1: stuck recovery is in flight — runs to completion. Must
            # come BEFORE ON_SLOPE so a slope-induced stuck recovery
            # doesn't get preempted by ON_SLOPE on every cooldown expiry.
            elif self._stuck_count > 30 or self._recovery_phase != 0:
                # GATE on fresh entry: if the rover has made strong recent
                # progress (>0.5 m gained in the last 2 s), treat the stuck
                # trigger as a false positive — clear and let BASE keep
                # going. Prevents "passed obstacle → flashback" reverse.
                do_recovery = True
                if self._recovery_phase == 0:
                    strong_progress = False
                    if len(self._stuck_history) >= 4:
                        ds_h = [h[3] for h in self._stuck_history
                                if not math.isnan(h[3])]
                        if len(ds_h) >= 4 and (ds_h[-4] - ds_h[-1]) > 0.5:
                            strong_progress = True
                    if strong_progress:
                        self._stuck_count = 0
                        self._interrupt_label = "BASE"
                        do_recovery = False
                    else:
                        self._recovery_phase = 1
                        self._recovery_step = 0
                        self._stuck_recovery_dir = 0.0
                        self._interrupt_steps = 0
                        self._interrupt_cmd = np.zeros(2, dtype=np.float32)
                        self._stuck_trigger_dist = float(
                            info.get("distance_to_target", 0.0))
                        self._interrupt_cooldown.pop("ON_SLOPE", None)
                # Abort recovery early if rover has already passed obstacle.
                if do_recovery:
                    cur_d_rec = float(info.get("distance_to_target", 0.0))
                    passed = (self._stuck_trigger_dist is not None
                              and cur_d_rec + self._post_commit_pass_margin_m
                                  < self._stuck_trigger_dist)
                    if passed:
                        self._recovery_phase = 0
                        self._recovery_step = 0
                        self._stuck_recovery_dir = 0.0
                        self._stuck_count = 0
                        self._stuck_trigger_dist = None
                        self._interrupt_label = "STUCK_ABORTED"
                        do_recovery = False
                if do_recovery:
                    cmd = self._stuck_recovery_cmd()
                    self._recovery_step += 1
                    if cmd is None:
                        # Phase 4 cooldown: yield to BASE but freeze stuck/
                        # ON_SLOPE re-trigger for the cooldown duration.
                        if self._recovery_step >= 50:
                            self._recovery_phase = 0
                            self._recovery_step = 0
                            self._stuck_recovery_dir = 0.0
                        self._interrupt_label = "RECOVERY_COOLDOWN"
                    else:
                        cmd_action_np = cmd
                        self._interrupt_label = f"STUCK_PHASE{self._recovery_phase}"
            # P0.5: rover on a slope. Trigger combines IMU tilt (primary)
            # with BEV rough/elev rise (secondary). If IMU sees high tilt
            # OR BEV sees clearly bad terrain ahead (with non-trivial IMU
            # tilt), fire ON_SLOPE. This uses both sensors instead of one.
            elif self._on_slope_combined_trigger(obs, tilt_now) \
                    and "ON_SLOPE" not in self._interrupt_cooldown:
                try:
                    q = odom_msg.pose.pose.orientation
                    roll_v, pitch_v, _ = quat_to_rpy(q.x, q.y, q.z, q.w)
                except Exception:
                    roll_v, pitch_v = 0.0, 0.0
                # Only pick a new direction the FIRST time we go on slope.
                # If _slope_turn_dir is already set, we're still on the
                # same slope event — keep the same side.
                if self._slope_turn_dir == 0.0:
                    if abs(roll_v) > 0.1:
                        # Rolled to one side already → turn opposite of
                        # the lean (downhill perpendicular).
                        self._slope_turn_dir = -1.0 if roll_v > 0 else 1.0
                    else:
                        # Pure pitch (head-on slope) → consult BEV slope
                        # distribution; turn toward the side with fewer
                        # bad rays. Fall back to goal bearing only when
                        # BEV has no useful signal (both sides clean).
                        _, _, dir_left, dir_right = self._bev_side_scores(obs)
                        if abs(dir_left) < 1e-3 and abs(dir_right) < 1e-3:
                            bearing_to_goal = float(info.get(
                                "bearing_to_target", 0.0))
                            self._slope_turn_dir = (
                                1.0 if bearing_to_goal >= 0 else -1.0)
                        elif abs(dir_left - dir_right) < 0.05:
                            # Near-tie → alternate with previous slope
                            # event so the rover explores both sides.
                            self._slope_turn_dir = (
                                -self._last_slope_dir
                                if self._last_slope_dir != 0.0
                                else 1.0)
                        else:
                            # Turn toward the lower-score (smoother +
                            # lower elevation rise) side.
                            self._slope_turn_dir = (
                                1.0 if dir_left < dir_right else -1.0)
                # Rotate briefly (~12 steps ≈ 55°) then yield to BASE for
                # ~80 steps (8 s) so the rover actually drives forward and
                # tests whether the new heading is better. Pure rotation
                # on a slope keeps tilt magnitude constant — so we MUST
                # leave time for translation between rotations.
                self._interrupt_cooldown["ON_SLOPE"] = 90
                cmd_action_np = np.array(
                    [0.0, 0.8 * self._slope_turn_dir], dtype=np.float32)
                self._interrupt_cmd = cmd_action_np.copy()
                # Record distance at trigger so the post-slope commit can
                # abort early once we've made progress past the slope.
                self._slope_trigger_dist = float(
                    info.get("distance_to_target", 0.0))
                self._interrupt_steps = 12
                self._interrupt_label = "ON_SLOPE"
            # P2: just escaped stuck — straight forward to clear area.
            # Aborts if rover is already closer to goal than when stuck
            # was triggered (means obstacle is behind us, no need to
            # blindly drive forward — let BASE re-target).
            elif self._post_stuck_commit_steps > 0:
                cur_d = float(info.get("distance_to_target", 0.0))
                if (self._stuck_trigger_dist is not None
                        and cur_d + self._post_commit_pass_margin_m
                            < self._stuck_trigger_dist):
                    # Obstacle passed — abort commit.
                    self._post_stuck_commit_steps = 0
                    self._stuck_trigger_dist = None
                else:
                    cmd_action_np = np.array([0.4, 0.0], dtype=np.float32)
                    self._post_stuck_commit_steps -= 1
                    self._interrupt_label = "POST_STUCK"
            # P3: an avoidance interrupt is still running from a prior
            # frame — hold its cmd for its full duration so the rover
            # commits to the maneuver instead of flickering decisions.
            # When the held interrupt is ON_SLOPE and finishes this step,
            # schedule a post-slope commit so the new heading actually
            # gets to translate forward before BASE re-engages.
            elif self._interrupt_steps > 0:
                cmd_action_np = self._interrupt_cmd.copy()
                self._interrupt_steps -= 1
                if (self._interrupt_steps == 0
                        and self._interrupt_label == "ON_SLOPE"):
                    self._post_slope_commit_steps = self._post_slope_commit_duration
            # P3.5: post-slope forward commit (aborts if obstacle passed).
            elif self._post_slope_commit_steps > 0:
                cur_d = float(info.get("distance_to_target", 0.0))
                if (self._slope_trigger_dist is not None
                        and cur_d + self._post_commit_pass_margin_m
                            < self._slope_trigger_dist):
                    self._post_slope_commit_steps = 0
                    self._slope_trigger_dist = None
                else:
                    cmd_action_np = np.array([0.4, 0.0], dtype=np.float32)
                    self._post_slope_commit_steps -= 1
                    self._interrupt_label = "POST_SLOPE"
            # P4: fresh BEV-based detection. Order inside detector:
            #     LOST_VISION (cliff blind spot) > NEG_OBSTACLE > HIGH_SLOPE.
            else:
                ev = self._detect_terrain_interrupt(obs)
                if ev is not None:
                    cmd_action_np = ev["cmd"]
                    self._interrupt_cmd = ev["cmd"].copy()
                    self._interrupt_steps = ev["duration"]
                    self._interrupt_label = ev["label"]
                    # RETRACE_RETURN trigger: count terrain interrupts
                    # during RETURN phase. If multiple fire in the recent
                    # window, switch to waypoint retrace for this episode.
                    if (info.get("phase") and
                            info["phase"].name == "RETURN" and
                            ev["label"] in ("NEG_OBSTACLE", "HIGH_SLOPE")):
                        self._return_avoid_hits += 1
                        if (not self._retrace_mode
                                and self._return_avoid_hits
                                >= self._return_avoid_trigger):
                            self._retrace_mode = True
                            self.get_logger().info(
                                f"[interrupt] RETRACE_RETURN engaged after "
                                f"{self._return_avoid_hits} terrain hits")
                else:
                    self._interrupt_label = "BASE"
            # Tick down cooldowns regardless.
            for k in list(self._interrupt_cooldown.keys()):
                self._interrupt_cooldown[k] -= 1
                if self._interrupt_cooldown[k] <= 0:
                    del self._interrupt_cooldown[k]
            # ABLATE_INTERRUPTS: undo all overrides applied above.
            # The interrupts still ran (so their internal state stays
            # consistent if the ablation is toggled mid-experiment),
            # but the published cmd reverts to the actor-derived one.
            if ABLATE_INTERRUPTS:
                cmd_action_np = _actor_cmd_snapshot
                self._interrupt_label = "BASE"
            # Decay the RETRACE counter slowly — old hits fade out.
            if self.episode_steps % self._return_avoid_window_steps == 0:
                self._return_avoid_hits = max(0, self._return_avoid_hits - 1)

        if self.last_obs is not None:
            reward, done = self._reward_and_done(obs, info)
            transition = Transition(
                obs=self.last_obs,
                action=torch.from_numpy(self.prev_action.copy()),
                reward=reward,
                done=done,
                next_obs=obs,
                info={
                    "phase": info["phase"].name,
                    "distance_to_target": info["distance_to_target"],
                    "control_mode": control_mode_for_log,
                    "demo_type": "normal",
                    "source_failed_mission_id": None,
                },
            )
            if not HUMAN_DRIVE:
                self.buffer.add(transition)
            else:
                # Accumulate demo step; saved as NPZ on episode end.
                try:
                    self._human_demo_buffer.append({
                        "lidar_bev": self.last_obs["lidar_bev"].numpy()
                            if hasattr(self.last_obs.get("lidar_bev"), "numpy")
                            else None,
                        "imu_vec": self.last_obs["imu_vec"].numpy()
                            if hasattr(self.last_obs.get("imu_vec"), "numpy")
                            else None,
                        "goal_vector": self.last_obs["goal_vector"].numpy()
                            if hasattr(self.last_obs.get("goal_vector"), "numpy")
                            else None,
                        "action": self.prev_action.copy().astype(np.float32),
                        "reward": float(reward),
                        "phase": info["phase"].name,
                    })
                except Exception:
                    pass
            self.total_env_steps += 1

            if done:
                self._publish_stop()
                # HUMAN_DRIVE: save the per-episode demo buffer to NPZ.
                if HUMAN_DRIVE and self._human_demo_buffer:
                    try:
                        from pathlib import Path as _P
                        demos_dir = _P("demos"); demos_dir.mkdir(exist_ok=True)
                        idx = self._human_demo_path_idx
                        while (demos_dir / f"demo_{idx:03d}.npz").exists():
                            idx += 1
                        self._human_demo_path_idx = idx + 1
                        d = self._human_demo_buffer
                        np.savez(
                            demos_dir / f"demo_{idx:03d}.npz",
                            lidar_bev=np.stack([s["lidar_bev"] for s in d if s["lidar_bev"] is not None]),
                            imu_vec=np.stack([s["imu_vec"] for s in d if s["imu_vec"] is not None]),
                            goal_vector=np.stack([s["goal_vector"] for s in d if s["goal_vector"] is not None]),
                            action=np.stack([s["action"] for s in d]),
                            reward=np.array([s["reward"] for s in d], dtype=np.float32),
                            phase=np.array([s["phase"] for s in d]),
                        )
                        self.get_logger().info(
                            f"[human-drive] saved demo_{idx:03d}.npz ({len(d)} steps)")
                    except Exception as e:
                        self.get_logger().warn(f"[human-drive] demo save failed: {e}")
                    self._human_demo_buffer = []
                # Episode end (success / tilt / timeout) → reset on next step.
                # Beyond the active flag, sweep the world right now: any stale
                # flag from a previous episode that the 6-retry budget couldn't
                # remove will be re-enqueued so the new episode starts clean.
                self._episode_active = False
                self._remove_active_flag()
                self._enqueue_stale_goal_flags()
                # Reset per-episode state so prior commitments don't carry over.
                self._stuck_history = []
                self._stuck_count = 0
                self._post_stuck_commit_steps = 0
                self._cmd_prev = np.zeros(2, dtype=np.float32)
                self._theta_sub_prev = 0.0
                self._interrupt_steps = 0
                self._interrupt_cooldown = {}
                self._interrupt_label = "BASE"
                self._return_avoid_hits = 0
                self._retrace_mode = False
                self._stuck_recovery_dir = 0.0
                self._slope_turn_dir = 0.0
                self._slope_flat_counter = 0
                self._last_slope_dir = 0.0
                self._on_slope_tilt_high_steps = 0
                self._rollover_active_steps = 0
                self._tilt_prev = 0.0
                self._post_slope_commit_steps = 0
                self._recovery_phase = 0
                self._recovery_step = 0
                self._stuck_trigger_dist = None
                self._slope_trigger_dist = None
                return

        # Publish the low-level twist with a low-pass filter — suppresses
        # left/right sway when reactive sub-goal selection flips between
        # adjacent sweep candidates.
        a = self._cmd_smooth_alpha
        cmd_smoothed = a * cmd_action_np.astype(np.float32) \
                       + (1.0 - a) * self._cmd_prev
        self._cmd_prev = cmd_smoothed
        self._publish_cmd(cmd_smoothed)
        # Demo distillation: buffer stores the ACTUAL executed cmd
        # (reverse-mapped to the hierarchical action space). When an
        # interrupt is active, the executed cmd differs from the actor's
        # raw output; without this remap, the world model would learn
        # (s, actor_theta) → s' but s' was actually produced by the
        # interrupt cmd → mismatch. Reverse-mapping makes interrupts act
        # as on-policy demonstrations the actor can imitate via BC, so the
        # learned policy gradually internalises the avoidance behaviors
        # and the interrupt system becomes a teacher rather than a crutch.
        if ABLATE_DISTILLATION:
            # Ablation: store actor's raw action instead of reverse-
            # mapped executed cmd. This re-introduces the (s, a) -> s'
            # dynamics mismatch when interrupts override the cmd.
            self.prev_action = action_np.astype(np.float32)
        elif is_hierarchical:
            v_exe = float(cmd_smoothed[0])
            w_exe = float(cmd_smoothed[1])
            kp = 1.2  # FlagSeekingPolicyConfig.kp_omega
            # Reverse-map executed cmd to (theta_sub, r_sub):
            #   bearing_eff = ω / kp (clamped to [-π, π])
            #   r_eff sign carries direction (+forward, -reverse)
            #   r_eff magnitude = current distance to target (clamped 2..10)
            bearing_eff = max(-math.pi, min(math.pi, w_exe / kp))
            r_mag = max(2.0, min(10.0, float(info.get(
                "distance_to_target", 10.0))))
            if ABLATE_REVERSE:
                r_eff = r_mag  # disable reverse encoding (always positive)
            else:
                r_eff = -r_mag if v_exe < -0.05 else r_mag
            self.prev_action = np.array(
                [bearing_eff, r_eff], dtype=np.float32)
        else:
            self.prev_action = cmd_smoothed.astype(np.float32)
        self.last_obs = obs
        self.last_distance_to_target = info["distance_to_target"]
        self.episode_steps += 1

        now = time.time()
        if now - self._last_checkpoint > CHECKPOINT_INTERVAL_S:
            self._save_checkpoint()
            self._last_checkpoint = now

    # ──────────────────────────────────────────────────────────────────────
    # Training worker
    # ──────────────────────────────────────────────────────────────────────
    def _train_loop(self):
        torch.set_num_threads(1)
        while not self._train_stop.is_set():
            if not self.buffer.can_sample(
                self.cfg.train.batch_size, self.cfg.train.seq_len
            ) or self.total_env_steps < self.cfg.train.min_replay:
                time.sleep(0.5)
                continue
            try:
                metrics = self.trainer.train_step(self.buffer)
            except Exception as exc:
                self.get_logger().warn(f"trainer step failed: {exc}")
                time.sleep(1.0)
                continue
            if metrics is None:
                time.sleep(0.1)
                continue
            self.total_train_steps += 1
            with self._metrics_lock:
                self._latest_metrics = metrics
            time.sleep(0.02)

    # ──────────────────────────────────────────────────────────────────────
    # Logging + checkpoints
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
            msg += " | " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        self.get_logger().info(msg)

    def _save_checkpoint(self, keep_n: int = 5):
        path = CHECKPOINT_DIR / f"td_step_{self.total_train_steps:07d}.pt"
        try:
            self.trainer.save_checkpoint(str(path))
            self.get_logger().info(f"checkpoint -> {path}")
        except Exception as exc:
            self.get_logger().warn(f"checkpoint save failed: {exc}")
            return
        try:
            ckpts = sorted(CHECKPOINT_DIR.glob("td_step_*.pt"))
            for old in ckpts[:-keep_n]:
                old.unlink(missing_ok=True)
        except Exception as exc:
            self.get_logger().warn(f"checkpoint cleanup failed: {exc}")

    def shutdown(self):
        self._train_stop.set()
        self._gt_thread_run = False
        self._publish_stop()
        self._remove_active_flag()
        # Give the cleanup thread a moment to drain its queue, then stop it.
        try:
            for _ in range(20):
                if self._pending_removals.empty():
                    break
                time.sleep(0.5)
        except Exception:
            pass
        self._cleanup_thread_run = False
        try:
            self._save_checkpoint()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = CraterTrainNode()
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
