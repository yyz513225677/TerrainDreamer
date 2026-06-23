"""Hyperparameters for the lunar DreamerV3 navigation algorithm.

Pure dataclass; no torch imports needed at module load.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    # Observation shapes
    bev_shape: Tuple[int, int, int] = (4, 128, 128)   # (C, H, W)
    imu_dim: int = 12
    goal_dim: int = 4
    mission_phase_dim: int = 2
    action_dim: int = 2

    # Encoder output sizes
    terrain_embed: int = 512
    imu_embed: int = 128
    goal_embed: int = 128
    obs_embed: int = 768   # fused observation embedding

    # RSSM
    rssm_hidden: int = 1024     # deterministic GRU hidden / state
    rssm_stoch: int = 32        # stochastic latent size (continuous Gaussian)
    rssm_min_std: float = 0.1   # stability floor for posterior/prior std

    # Reward / continuation
    reward_hidden: int = 256
    cont_hidden: int = 256

    # Actor-critic
    actor_hidden: int = 512
    critic_hidden: int = 512
    action_low: Tuple[float, float] = (0.0, -1.0)    # (linear_x, angular_z)
    action_high: Tuple[float, float] = (0.8, 1.0)
    actor_min_std: float = 0.1


@dataclass
class MissionConfig:
    waypoint_spacing_m: float = 0.5     # save outbound waypoints every 0.5 m
    waypoint_reach_m: float = 0.5       # pop a return waypoint when this close
    destination_reach_m: float = 0.8    # OUTBOUND -> RETURN trigger
    origin_reach_m: float = 0.8         # success trigger on RETURN


@dataclass
class SafetyConfig:
    lidar_brake_dist: float = 1.5
    lidar_stop_dist: float = 0.6
    brake_linear_scale_min: float = 0.0
    avoid_angular_kick: float = 0.5


@dataclass
class RewardConfig:
    progress_weight: float = 1.0
    success_bonus: float = 10.0
    collision_penalty: float = -10.0
    step_penalty: float = -0.01
    angular_penalty: float = -0.02


@dataclass
class TrainConfig:
    batch_size: int = 16
    seq_len: int = 50
    imagine_horizon: int = 15
    replay_capacity: int = 100_000
    min_replay: int = 1_000

    world_lr: float = 1e-4
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5

    discount: float = 0.997
    lambda_gae: float = 0.95
    kl_balance: float = 0.8          # DreamerV3 KL balance
    free_nats: float = 1.0
    grad_clip: float = 1000.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cuda"

    @property
    def feat_dim(self) -> int:
        return self.model.rssm_hidden + self.model.rssm_stoch
