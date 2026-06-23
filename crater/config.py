"""TerrainDreamer configuration dataclasses.

Pure-Python configuration; no torch imports needed at module load.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    # Observation shapes
    bev_shape: Tuple[int, int, int] = (4, 128, 128)
    imu_dim: int = 12
    goal_dim: int = 4
    mission_phase_dim: int = 2
    action_dim: int = 2

    # Encoder output sizes
    terrain_embed: int = 512
    imu_embed: int = 128
    goal_embed: int = 128
    obs_embed: int = 768

    # RSSM
    rssm_hidden: int = 1024
    rssm_stoch: int = 32
    rssm_min_std: float = 0.1

    reward_hidden: int = 256
    cont_hidden: int = 256

    actor_hidden: int = 512
    critic_hidden: int = 512
    # Hierarchical sub-goal action space (polar coords in rover frame):
    #   action[0] = theta_sub ∈ [-pi, pi]      (bearing toward sub-goal)
    #   action[1] = r_sub     ∈ [r_min, r_max] (range)
    # The driver converts (theta, r) → world-frame waypoint, and the
    # FlagSeekingPolicy generates the low-level [v, omega] command toward it.
    # r_sub allowed negative: sign encodes direction (positive = forward,
    # negative = reverse). Magnitude is look-ahead distance.
    action_low: Tuple[float, float] = (-math.pi, -10.0)
    action_high: Tuple[float, float] = (math.pi, 10.0)
    actor_min_std: float = 0.1
    # When True, the driver interprets actor output as a sub-goal in the
    # rover's body frame and delegates motor control to the FlagSeekingPolicy.
    # When False, actor output is consumed directly as [linear_x, angular_z].
    use_hierarchical_action: bool = True


@dataclass
class MissionConfig:
    waypoint_spacing_m: float = 0.5      # save outbound waypoint every 0.5 m
    waypoint_reach_m: float = 0.5        # pop a return waypoint when within 0.5 m
    destination_reach_m: float = 0.8     # OUTBOUND -> RETURN trigger
    origin_reach_m: float = 0.8          # SUCCESS trigger on RETURN


@dataclass
class SafetyConfig:
    lidar_brake_dist: float = 1.5
    lidar_stop_dist: float = 0.6
    brake_linear_scale_min: float = 0.0
    avoid_angular_kick: float = 0.5

    # Tilt thresholds (radians)
    tilt_brake_rad: float = 0.45         # start slowing down at ~26°
    tilt_stop_rad: float = 0.80          # near-flip — full stop and reorient

    # Obstacle risk on BEV occupancy channel
    obstacle_risk_radius_m: float = 2.0  # forward-cone radius for risk eval
    obstacle_risk_threshold: float = 0.20 # fraction of occupied cells -> risky
    obstacle_stop_threshold: float = 0.45 # fraction occupied -> emergency stop


@dataclass
class RewardConfig:
    progress_weight: float = 1.0
    success_bonus: float = 10.0
    collision_penalty: float = -10.0
    step_penalty: float = -0.01
    angular_penalty: float = -0.02
    # Spatial-shaping penalties are DISABLED by default in the
    # demonstration-anchored regime: we no longer hand-craft obstacle
    # avoidance into the reward; obstacle awareness is supposed to come
    # entirely from imitating expert demos (BC) plus the latent
    # traversability head learnt via imagined-reward shaping.
    tilt_penalty_weight: float = 0.0
    tilt_penalty_threshold_rad: float = 0.30
    lidar_penalty_weight: float = 0.0
    lidar_penalty_threshold: float = 0.70
    timeout_penalty: float = -5.0              # task-level, not spatial


@dataclass
class TrainConfig:
    batch_size: int = 16
    seq_len: int = 50
    imagine_horizon: int = 15
    replay_capacity: int = 250_000
    min_replay: int = 1_000

    world_lr: float = 1e-4
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5

    discount: float = 0.997
    lambda_gae: float = 0.95
    kl_balance: float = 0.8
    free_nats: float = 1.0
    grad_clip: float = 1000.0


@dataclass
class HumanTakeoverConfig:
    enable_human_takeover: bool = True
    default_mode: str = "autonomous"        # 'autonomous' | 'human'
    manual_control_timeout: float = 1.0     # seconds; if human stops sending,
                                            # fall back to autonomous


@dataclass
class BCConfig:
    # Demonstration-anchored regime: BC is the dominant actor signal, and
    # the standard Dreamer actor-loss (return-maximising) is down-weighted
    # so it acts more like a *regulariser* than the primary objective.
    behavior_cloning_weight: float = 2.0
    recovery_behavior_cloning_weight: float = 4.0     # higher than BC
    dreamer_actor_weight: float = 0.3                 # scales the imag-return loss
    bc_loss_type: str = "mse"               # 'mse' | 'l1'


@dataclass
class TraversabilityConfig:
    """Learned traversability head + adaptive reward normalization.

    Replaces hand-tuned lidar/tilt penalties with a head that learns
    P(traversable | latent_state) from observed tilt + lidar signals. In
    imagined rollouts the head value softly shapes the reward, with a
    Lagrangian-style dual variable adapting the weight so the actor sees
    a stable cost signal regardless of reward scale.
    """
    enable: bool = False                      # off by default; explicit opt-in
    head_hidden: int = 256
    head_depth: int = 2
    label_tilt_sigma: float = 0.35            # rad
    label_lidar_sigma: float = 0.35
    # Demonstration-anchored regime: set head_loss_weight to 0 so the
    # traversability head has no synthetic-label supervision; it only
    # learns via the actor gradient feedback inside imagined rollouts.
    head_loss_weight: float = 0.0
    head_lr: float = 1e-4

    # How much the traversability term contributes to imagined reward at
    # initialisation. The dual variable adapts this up/down.
    init_lambda: float = 0.5
    lambda_lr: float = 0.01
    target_traversable_frac: float = 0.85     # actor should hit ≥ 85 %
    lambda_min: float = 0.0
    lambda_max: float = 5.0

    # Return normalization (DreamerV3 percentile trick) — independent of
    # traversability but enabled together since the same calibration is
    # what makes the trav signal stable.
    enable_return_norm: bool = True
    return_ema_decay: float = 0.99
    return_norm_eps: float = 1.0              # min denominator (Dreamer paper)


@dataclass
class FailedMissionConfig:
    enable_failed_mission_memory: bool = True
    max_failed_missions: int = 256
    enable_human_recovery_learning: bool = True
    failed_mission_save_path: str = "failed_missions.pt"
    recovery_demo_priority: float = 0.5     # fraction of mixed-batch slots
                                            # dedicated to recovery demos


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    human: HumanTakeoverConfig = field(default_factory=HumanTakeoverConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    failed: FailedMissionConfig = field(default_factory=FailedMissionConfig)
    trav: TraversabilityConfig = field(default_factory=TraversabilityConfig)
    device: str = "cuda"

    @property
    def feat_dim(self) -> int:
        return self.model.rssm_hidden + self.model.rssm_stoch
