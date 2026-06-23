"""Dreamer-style training scaffold.

This file is a clean *scaffold*: it lays out the three update steps
(world model, actor, critic) with their gradients flowing the right way
and connects every model component, but leaves the exact DreamerV3 loss
recipe (symlog two-hot returns, EMA critic, percentile return norm,
etc.) marked with TODOs. Even as-is it should be runnable end-to-end
on synthetic data — useful for sanity-checking the wiring before
plumbing in a ROS 2 environment.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .model import LunarDreamer, Obs
from .replay_buffer import ReplayBuffer
from .rssm import RSSM, RSSMState


class DreamerTrainer:
    def __init__(self, model: LunarDreamer, cfg: Config):
        self.model = model
        self.cfg = cfg
        device = torch.device(
            cfg.device if (cfg.device == "cpu" or torch.cuda.is_available())
            else "cpu")
        self.device = device
        self.model.to(device)

        t = cfg.train
        self.opt_world = torch.optim.Adam(
            list(self.model.world_model_parameters()), lr=t.world_lr)
        self.opt_actor = torch.optim.Adam(
            self.model.actor_parameters(), lr=t.actor_lr)
        self.opt_critic = torch.optim.Adam(
            self.model.critic_parameters(), lr=t.critic_lr)

    # ──────────────────────────────────────────────────────────────────────
    # Convenience batching
    # ──────────────────────────────────────────────────────────────────────
    def _move(self, batch):
        obs, actions, rewards, dones, _ = batch
        obs = {k: v.to(self.device) for k, v in obs.items()}
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        return obs, actions, rewards, dones

    # ──────────────────────────────────────────────────────────────────────
    # World-model update — encode obs, run posterior, score reward/cont/KL
    # ──────────────────────────────────────────────────────────────────────
    def update_world_model(
        self,
        obs: Obs,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[Dict[str, float], RSSMState]:
        cfg = self.cfg.train

        # Encode obs to [B, T, obs_embed].
        embeds = self.model.encode_obs_seq(obs)
        prior, post = self.model.rssm.observe(embeds, actions)
        feats = RSSM.get_feat(post)              # [B, T, feat_dim]

        # Predicted reward + continuation from posterior feats.
        pred_reward = self.model.reward_head(feats)
        cont_logits = self.model.continuation_head(feats)
        cont_target = 1.0 - dones

        # TODO(dreamerv3): swap MSE for symlog two-hot regression on returns.
        loss_reward = F.mse_loss(pred_reward, rewards)
        loss_cont = F.binary_cross_entropy_with_logits(cont_logits, cont_target)
        loss_kl = self.model.rssm.kl_loss(
            prior, post, free_nats=cfg.free_nats, balance=cfg.kl_balance)

        # TODO(dreamerv3): add observation reconstruction losses for
        # BEV/IMU/goal so the latent stays grounded.
        loss = loss_reward + loss_cont + loss_kl

        self.opt_world.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.world_model_parameters()), cfg.grad_clip)
        self.opt_world.step()

        metrics = {
            "loss/world_total": float(loss.detach()),
            "loss/reward": float(loss_reward.detach()),
            "loss/cont": float(loss_cont.detach()),
            "loss/kl": float(loss_kl.detach()),
        }
        # Final posterior state, detached, is the starting point for the
        # imagination rollout below.
        last = RSSMState(
            post.h[:, -1].detach(),
            post.z[:, -1].detach(),
            post.mean[:, -1].detach(),
            post.std[:, -1].detach(),
        )
        return metrics, last

    # ──────────────────────────────────────────────────────────────────────
    # Actor / critic update via imagined rollouts
    # ──────────────────────────────────────────────────────────────────────
    def update_actor_critic(self, init: RSSMState) -> Dict[str, float]:
        cfg = self.cfg.train

        def policy(feat: torch.Tensor) -> torch.Tensor:
            _, action = self.model.actor(feat)
            return action

        states, actions = self.model.rssm.imagine(init, policy, cfg.imagine_horizon)
        feats = RSSM.get_feat(states)            # [B, H, feat_dim]
        with torch.no_grad():
            # TODO(dreamerv3): use EMA target critic + return normalization.
            values = self.model.critic(feats)
            rewards = self.model.reward_head(feats)
            cont = torch.sigmoid(self.model.continuation_head(feats))

        # λ-returns over the imagined horizon (vanilla TD(λ)).
        returns = self._lambda_return(rewards, values, cont, cfg.discount, cfg.lambda_gae)

        # Actor: maximize returns. Use re-evaluated policy for grad flow.
        feats_grad = RSSM.get_feat(states)
        dist, fresh_actions = self.model.actor(feats_grad)
        log_prob = self.model.actor.log_prob(dist, fresh_actions)
        # TODO(dreamerv3): subtract baseline (target critic value) and
        # normalize by 5th-95th percentile return spread.
        advantage = (returns - values).detach()
        actor_loss = -(log_prob * advantage).mean()

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            list(self.model.actor_parameters()), cfg.grad_clip)
        self.opt_actor.step()

        # Critic: regress predicted value onto returns.
        value_pred = self.model.critic(feats_grad.detach())
        critic_loss = F.mse_loss(value_pred, returns.detach())

        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.critic_parameters()), cfg.grad_clip)
        self.opt_critic.step()

        return {
            "loss/actor": float(actor_loss.detach()),
            "loss/critic": float(critic_loss.detach()),
            "stat/imag_return_mean": float(returns.mean().detach()),
        }

    # ──────────────────────────────────────────────────────────────────────
    # One full Dreamer training step
    # ──────────────────────────────────────────────────────────────────────
    def step(self, buffer: ReplayBuffer) -> Optional[Dict[str, float]]:
        cfg = self.cfg.train
        if not buffer.can_sample(cfg.batch_size, cfg.seq_len):
            return None
        batch = buffer.sample(cfg.batch_size, cfg.seq_len)
        obs, actions, rewards, dones = self._move(batch)
        world_metrics, last_state = self.update_world_model(
            obs, actions, rewards, dones)
        ac_metrics = self.update_actor_critic(last_state)
        return {**world_metrics, **ac_metrics}

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _lambda_return(
        rewards: torch.Tensor,    # [B, H]
        values: torch.Tensor,     # [B, H]
        cont: torch.Tensor,       # [B, H]
        discount: float,
        lam: float,
    ) -> torch.Tensor:
        """Generalized advantage / TD(λ) target over an imagined horizon."""
        B, H = rewards.shape
        returns = torch.zeros_like(rewards)
        next_return = values[:, -1]
        for t in reversed(range(H)):
            bootstrap = (1 - lam) * values[:, t] + lam * next_return
            returns[:, t] = rewards[:, t] + discount * cont[:, t] * bootstrap
            next_return = returns[:, t]
        return returns

    def save(self, path: str) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "opt_world": self.opt_world.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt_world.load_state_dict(ckpt["opt_world"])
        self.opt_actor.load_state_dict(ckpt["opt_actor"])
        self.opt_critic.load_state_dict(ckpt["opt_critic"])
