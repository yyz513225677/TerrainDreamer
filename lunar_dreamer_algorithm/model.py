"""Top-level DreamerV3 model — wires encoders + RSSM + heads + actor/critic.

Inference path (single step):
    obs  -> encoders -> fused embed -> RSSM.observe_step (posterior) ->
    feat -> actor.act -> shielded action

The full Dreamer training loop lives in `trainer.py`.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .actor_critic import Actor, Critic
from .config import Config
from .continuation_model import ContinuationModel
from .fusion_encoder import FusionEncoder
from .goal_encoder import GoalEncoder
from .imu_encoder import IMUEncoder
from .reward_model import RewardModel
from .rssm import RSSM, RSSMState
from .terrain_encoder import TerrainEncoder


Obs = Dict[str, torch.Tensor]


class LunarDreamer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        m = cfg.model

        self.terrain_encoder = TerrainEncoder(
            in_channels=m.bev_shape[0], embed_dim=m.terrain_embed)
        self.imu_encoder = IMUEncoder(
            imu_dim=m.imu_dim, embed_dim=m.imu_embed)
        self.goal_encoder = GoalEncoder(
            goal_dim=m.goal_dim, phase_dim=m.mission_phase_dim,
            action_dim=m.action_dim, embed_dim=m.goal_embed)
        self.fusion = FusionEncoder(
            terrain_embed=m.terrain_embed,
            imu_embed=m.imu_embed,
            goal_embed=m.goal_embed,
            obs_embed=m.obs_embed,
        )
        self.rssm = RSSM(
            action_dim=m.action_dim,
            obs_embed=m.obs_embed,
            hidden=m.rssm_hidden,
            stoch=m.rssm_stoch,
            min_std=m.rssm_min_std,
        )

        feat_dim = cfg.feat_dim
        self.reward_head = RewardModel(feat_dim, hidden=m.reward_hidden)
        self.continuation_head = ContinuationModel(feat_dim, hidden=m.cont_hidden)

        self.actor = Actor(
            feat_dim=feat_dim,
            action_dim=m.action_dim,
            hidden=m.actor_hidden,
            action_low=m.action_low,
            action_high=m.action_high,
            min_std=m.actor_min_std,
        )
        self.critic = Critic(feat_dim=feat_dim, hidden=m.critic_hidden)

    # ──────────────────────────────────────────────────────────────────────
    # Observation pipeline
    # ──────────────────────────────────────────────────────────────────────
    def encode_obs(self, obs: Obs) -> torch.Tensor:
        """obs fields all expected to be batched (leading B dim, or B*T)."""
        terrain = self.terrain_encoder(obs["bev"])
        imu = self.imu_encoder(obs["imu"])
        goal = self.goal_encoder(obs["goal"], obs["phase"], obs["prev_action"])
        return self.fusion(terrain, imu, goal)

    def encode_obs_seq(self, obs: Obs) -> torch.Tensor:
        """obs fields shape [B, T, ...] -> embed [B, T, obs_embed]."""
        B, T = obs["imu"].shape[:2]
        flat = {k: v.reshape(B * T, *v.shape[2:]) for k, v in obs.items()}
        embed = self.encode_obs(flat)
        return embed.view(B, T, -1)

    # ──────────────────────────────────────────────────────────────────────
    # Single-step inference (for ROS-side rollout)
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def act(
        self,
        obs: Obs,
        prev_state: Optional[RSSMState] = None,
        prev_action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, RSSMState]:
        device = next(self.parameters()).device
        embed = self.encode_obs(obs)
        B = embed.shape[0]
        if prev_state is None:
            prev_state = self.rssm.initial_state(B, device)
        if prev_action is None:
            prev_action = torch.zeros(B, self.cfg.model.action_dim, device=device)
        _, post = self.rssm.observe_step(prev_state, prev_action, embed)
        feat = RSSM.get_feat(post)
        action = self.actor.act(feat, deterministic=deterministic)
        return action, post

    # ──────────────────────────────────────────────────────────────────────
    # Module parameter groups (for the trainer's optimizers)
    # ──────────────────────────────────────────────────────────────────────
    def world_model_parameters(self):
        for mod in (
            self.terrain_encoder, self.imu_encoder, self.goal_encoder,
            self.fusion, self.rssm, self.reward_head, self.continuation_head,
        ):
            yield from mod.parameters()

    def actor_parameters(self):
        return self.actor.parameters()

    def critic_parameters(self):
        return self.critic.parameters()
