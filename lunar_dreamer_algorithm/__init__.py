"""lunar_dreamer_algorithm — DreamerV3-style navigation algorithm for a
lunar UGV. Pure PyTorch / Python; no ROS or Gazebo dependencies.

Typical wiring from an env wrapper:

    from lunar_dreamer_algorithm import (
        Config, LunarDreamer, DreamerTrainer, ReplayBuffer, Transition,
        MissionManager, MissionPhase, SafetyShield, TrajectoryMemory,
    )
"""
from __future__ import annotations

from .actor_critic import Actor, Critic
from .config import (
    Config,
    ModelConfig,
    MissionConfig,
    RewardConfig,
    SafetyConfig,
    TrainConfig,
)
from .continuation_model import ContinuationModel
from .fusion_encoder import FusionEncoder
from .goal_encoder import GoalEncoder
from .imu_encoder import IMUEncoder
from .mission_manager import MissionManager, MissionPhase, MissionStatus
from .model import LunarDreamer
from .obs_builders import (
    build_goal_vec,
    build_imu_vec,
    pack_obs,
    points_to_bev,
    quat_to_rpy,
)
from .replay_buffer import ReplayBuffer, Transition
from .reward_model import RewardModel
from .rssm import RSSM, RSSMState
from .safety_shield import SafetyShield
from .terrain_encoder import TerrainEncoder
from .trainer import DreamerTrainer
from .trajectory_memory import TrajectoryMemory

__all__ = [
    "Actor",
    "Config",
    "ContinuationModel",
    "Critic",
    "DreamerTrainer",
    "FusionEncoder",
    "GoalEncoder",
    "IMUEncoder",
    "LunarDreamer",
    "MissionConfig",
    "MissionManager",
    "MissionPhase",
    "MissionStatus",
    "ModelConfig",
    "ReplayBuffer",
    "RewardConfig",
    "RewardModel",
    "RSSM",
    "RSSMState",
    "SafetyConfig",
    "SafetyShield",
    "TerrainEncoder",
    "TrainConfig",
    "TrajectoryMemory",
    "Transition",
]
