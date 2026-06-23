"""CraterModel — top-level world-model + actor + safety/human stack.

Wires:
    BEVTerrainEncoder + IMUEncoder + GoalEncoder + FusionEncoder
    -> RSSM
    -> RewardModel, ContinuationModel
    -> ActorCritic
    + HumanTakeover, SafetyShield, FailedMissionBuffer

Inference path (single step):
    obs -> encode_obs -> RSSM.observe_step -> feat
         -> ActorCritic.sample_action -> SafetyShield.filter_action
         -> HumanTakeover.select_action  ->  final action

The full Dreamer + BC + recovery-BC training loop lives in `trainer.py`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .actor_critic import ActorCritic
from .bev_terrain_encoder import BEVTerrainEncoder
from .behavior_cloning import BehaviorCloning
from .config import Config
from .continuation_model import ContinuationModel
from .failed_mission_buffer import FailedMissionBuffer
from .fusion_encoder import FusionEncoder
from .goal_encoder import GoalEncoder
from .human_takeover import HumanTakeover, MODE_AUTONOMOUS, MODE_HUMAN
from .imu_encoder import IMUEncoder
from .reward_model import RewardModel
from .rssm import RSSM, State
from .safety_shield import SafetyShield


Obs = Dict[str, torch.Tensor]


class CraterModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        m = cfg.model

        # ── Encoders ──────────────────────────────────────────────────────
        self.bev_encoder = BEVTerrainEncoder(
            in_channels=m.bev_shape[0], embed_dim=m.terrain_embed)
        self.imu_encoder = IMUEncoder(
            imu_dim=m.imu_dim, embed_dim=m.imu_embed)
        self.goal_encoder = GoalEncoder(
            goal_dim=m.goal_dim, phase_dim=m.mission_phase_dim,
            action_dim=m.action_dim, embed_dim=m.goal_embed)
        self.fusion = FusionEncoder(
            terrain_embed=m.terrain_embed, imu_embed=m.imu_embed,
            goal_embed=m.goal_embed, obs_embed=m.obs_embed)

        # ── World model ────────────────────────────────────────────────────
        self.rssm = RSSM(
            action_dim=m.action_dim, obs_embed=m.obs_embed,
            hidden=m.rssm_hidden, stoch=m.rssm_stoch,
            min_std=m.rssm_min_std,
        )
        feat_dim = cfg.feat_dim
        self.reward_head = RewardModel(feat_dim, hidden=m.reward_hidden)
        self.continuation_head = ContinuationModel(feat_dim, hidden=m.cont_hidden)
        # Optional: learned traversability head — predicts P(traversable | feat).
        # When cfg.trav.enable is False this still exists (so checkpoints are
        # forward-compatible) but is never queried by the trainer.
        from .traversability_head import TraversabilityHead
        self.traversability_head = TraversabilityHead(
            feat_dim=feat_dim,
            hidden=cfg.trav.head_hidden,
            depth=cfg.trav.head_depth,
        )

        # ── Actor-critic ───────────────────────────────────────────────────
        self.actor_critic = ActorCritic(
            feat_dim=feat_dim,
            action_dim=m.action_dim,
            actor_hidden=m.actor_hidden,
            critic_hidden=m.critic_hidden,
            action_low=m.action_low,
            action_high=m.action_high,
            min_std=m.actor_min_std,
        )

        # ── Non-learnable runtime helpers ──────────────────────────────────
        self.safety = SafetyShield(
            cfg.safety,
            action_low=m.action_low,
            action_high=m.action_high,
        )
        self.takeover = HumanTakeover(
            default_mode=cfg.human.default_mode,
            manual_control_timeout=cfg.human.manual_control_timeout,
        )
        self.failed_missions = FailedMissionBuffer(
            max_failed_missions=cfg.failed.max_failed_missions,
        )
        self.bc = BehaviorCloning(loss_type=cfg.bc.bc_loss_type)

    # ──────────────────────────────────────────────────────────────────────
    # Observation pipeline
    # ──────────────────────────────────────────────────────────────────────
    def encode_obs(self, obs: Obs) -> torch.Tensor:
        """obs fields are batched (leading B dim or B*T)."""
        for k in ("lidar_bev", "imu", "goal_vector", "mission_phase", "prev_action"):
            if k not in obs:
                raise KeyError(
                    f"CraterModel.encode_obs: missing key '{k}' in obs")
        terrain = self.bev_encoder(obs["lidar_bev"])
        imu = self.imu_encoder(obs["imu"])
        goal = self.goal_encoder(
            obs["goal_vector"], obs["mission_phase"], obs["prev_action"])
        return self.fusion(terrain, imu, goal)

    def encode_obs_seq(self, obs: Obs) -> torch.Tensor:
        """obs fields shape [B, T, ...] -> embed [B, T, obs_embed]."""
        B, T = obs["imu"].shape[:2]
        flat = {k: v.reshape(B * T, *v.shape[2:]) for k, v in obs.items()}
        return self.encode_obs(flat).view(B, T, -1)

    # ──────────────────────────────────────────────────────────────────────
    # Single-step inference
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def act(
        self,
        obs: Obs,
        state: Optional[State] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, State]:
        """Run encoder → RSSM posterior step → actor → safety shield. Returns
        (shielded action, posterior state)."""
        device = next(self.parameters()).device
        embed = self.encode_obs(obs)
        B = embed.shape[0]
        if state is None:
            state = self.rssm.init_state(B, device)
        prev_action = obs.get("prev_action", torch.zeros(B, self.cfg.model.action_dim,
                                                         device=device))
        _, post = self.rssm.observe_step(state, prev_action, embed)
        feat = RSSM.get_feat(post)
        action, _, _ = self.actor_critic.sample_action(feat, deterministic=deterministic)
        shielded = self.safety.filter_action(action, obs)
        return shielded, post

    def select_action(
        self,
        obs: Obs,
        state: Optional[State] = None,
        human_action: Optional[torch.Tensor] = None,
        control_mode: str = MODE_AUTONOMOUS,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, State]:
        """Same as `act` but routed through HumanTakeover so a human can
        override the autonomous action when `control_mode == 'human'`."""
        self.takeover.set_mode(control_mode)
        auto_action, new_state = self.act(obs, state=state, deterministic=deterministic)
        final = self.takeover.select_action(auto_action, human_action=human_action)
        return final, new_state

    # ──────────────────────────────────────────────────────────────────────
    # Imagination (for the trainer's actor/critic update)
    # ──────────────────────────────────────────────────────────────────────
    def imagine(self, state: State, horizon: int):
        def policy(feat: torch.Tensor) -> torch.Tensor:
            action, _, _ = self.actor_critic.sample_action(feat)
            return action
        return self.rssm.imagine(state, policy, horizon)

    # ──────────────────────────────────────────────────────────────────────
    # Failed-mission + recovery hooks (delegate to FailedMissionBuffer)
    # ──────────────────────────────────────────────────────────────────────
    def record_failed_mission(self, mission_data: Dict[str, Any]) -> int:
        return self.failed_missions.add_failed_mission(mission_data)

    def record_recovery_demo(
        self,
        replay_buffer,            # ReplayBuffer
        obs: Obs,
        human_action: torch.Tensor,
        reward: float,
        next_obs: Obs,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Stash a single recovery-demo transition into the replay buffer
        with the right metadata so the trainer can sample it for the
        recovery-BC loss."""
        from .replay_buffer import Transition, CONTROL_MODE_HUMAN, DEMO_TYPE_RECOVERY
        info = dict(info) if info else {}
        info["control_mode"] = CONTROL_MODE_HUMAN
        info["demo_type"] = DEMO_TYPE_RECOVERY
        replay_buffer.add(Transition(obs, human_action, reward, done, next_obs, info))

    # ──────────────────────────────────────────────────────────────────────
    # Parameter groups (for Trainer's optimizers)
    # ──────────────────────────────────────────────────────────────────────
    def world_model_parameters(self):
        for mod in (
            self.bev_encoder, self.imu_encoder, self.goal_encoder,
            self.fusion, self.rssm, self.reward_head, self.continuation_head,
            self.traversability_head,
        ):
            yield from mod.parameters()

    def actor_parameters(self):
        # ActorCritic shares one Module — split by tensor namespace.
        for n, p in self.actor_critic.named_parameters():
            if n.startswith("actor_"):
                yield p

    def critic_parameters(self):
        for n, p in self.actor_critic.named_parameters():
            if n.startswith("critic_"):
                yield p

    # ──────────────────────────────────────────────────────────────────────
    # Save / load
    # ──────────────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict()}, path)

    def load(self, path: str, map_location: str = "cpu") -> None:
        ckpt = torch.load(path, map_location=map_location)
        self.load_state_dict(ckpt["state_dict"])
