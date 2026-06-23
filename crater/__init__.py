"""crater (TerrainDreamer — Cost-Reasoning Adaptive TERrain navigator) — pure PyTorch algorithm for ROS 2 + Gazebo Sim lunar
Jackal autonomous navigation. No ROS / Gazebo imports anywhere in this
module — the ROS 2 Gymnasium wrapper is expected to provide already-processed
observations and route human-takeover actions / failed-mission metadata in.
"""
from __future__ import annotations

from .actor_critic import ActorCritic
from .base_policy import FlagSeekingPolicy, FlagSeekingPolicyConfig
from .behavior_cloning import BehaviorCloning
from .bev_terrain_encoder import BEVTerrainEncoder
from .config import (
    BCConfig,
    Config,
    FailedMissionConfig,
    HumanTakeoverConfig,
    MissionConfig,
    ModelConfig,
    RewardConfig,
    SafetyConfig,
    TrainConfig,
)
from .continuation_model import ContinuationModel
from .failed_mission_buffer import FAILURE_REASONS, FailedMissionBuffer
from .fusion_encoder import FusionEncoder
from .goal_encoder import GoalEncoder
from .human_takeover import MODE_AUTONOMOUS, MODE_HUMAN, HumanTakeover
from .imu_encoder import IMUEncoder
from .mission_manager import MissionManager, MissionPhase
from .model import CraterModel
from .replay_buffer import (
    CONTROL_MODE_AUTO,
    CONTROL_MODE_HUMAN,
    DEMO_TYPE_NORMAL,
    DEMO_TYPE_RECOVERY,
    ReplayBuffer,
    Transition,
)
from .reward_model import RewardModel
from .rssm import RSSM, State
from .traversability_head import TraversabilityHead, make_traversability_label
from .safety_shield import SafetyShield
from .trajectory_memory import TrajectoryMemory
from .trainer import Trainer

__all__ = [
    "ActorCritic",
    "BCConfig",
    "BEVTerrainEncoder",
    "BehaviorCloning",
    "CONTROL_MODE_AUTO",
    "CONTROL_MODE_HUMAN",
    "Config",
    "ContinuationModel",
    "DEMO_TYPE_NORMAL",
    "DEMO_TYPE_RECOVERY",
    "FAILURE_REASONS",
    "FailedMissionBuffer",
    "FailedMissionConfig",
    "FlagSeekingPolicy",
    "FlagSeekingPolicyConfig",
    "FusionEncoder",
    "GoalEncoder",
    "HumanTakeover",
    "HumanTakeoverConfig",
    "IMUEncoder",
    "MODE_AUTONOMOUS",
    "MODE_HUMAN",
    "MissionConfig",
    "MissionManager",
    "MissionPhase",
    "ModelConfig",
    "RSSM",
    "ReplayBuffer",
    "RewardConfig",
    "RewardModel",
    "SafetyConfig",
    "SafetyShield",
    "State",
    "CraterModel",
    "TrainConfig",
    "Trainer",
    "TraversabilityHead",
    "TrajectoryMemory",
    "Transition",
]
