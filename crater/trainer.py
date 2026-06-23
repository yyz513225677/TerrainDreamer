"""TerrainDreamer Trainer.

A clean Dreamer-style training scaffold that combines:

    total_actor_loss = dreamer_actor_loss
                       + behavior_cloning_weight       * bc_loss
                       + recovery_behavior_cloning_weight * recovery_bc_loss

Three optimisers (world / actor / critic) — independent gradient updates.
Marked TODOs indicate where the full DreamerV3 recipe (symlog two-hot
returns, EMA target critic, percentile return normalisation, observation
reconstruction) should be wired in for production training.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import Config
from .model import CraterModel, Obs
from .replay_buffer import (
    CONTROL_MODE_HUMAN,
    DEMO_TYPE_RECOVERY,
    ReplayBuffer,
)
from .rssm import RSSM, State
from .traversability_head import make_traversability_label


class Trainer:
    def __init__(self, model: CraterModel, cfg: Config):
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
            list(self.model.actor_parameters()), lr=t.actor_lr)
        self.opt_critic = torch.optim.Adam(
            list(self.model.critic_parameters()), lr=t.critic_lr)

        # ── EMA target critic (DreamerV3 paper §4.3) ───────────────────────
        # The actor's λ-return targets are computed from the *target* critic
        # (slow-moving copy) instead of the online critic; this bounds value
        # estimation error and prevents the runaway critic spikes we saw
        # (max 21k in iter 4). EMA decay matches DreamerV3 paper.
        import copy
        # actor_critic.critic is a *method* — the actual module is critic_net.
        self._target_critic_net = copy.deepcopy(
            self.model.actor_critic.critic_net).to(device)
        for p in self._target_critic_net.parameters():
            p.requires_grad = False
        self._target_ema_decay = 0.98

        self._last_metrics: Dict[str, float] = {}

        # ── Adaptive reward normalisation (DreamerV3 percentile trick) ──
        # Tracks EMA of return p5 / p95 so the actor sees a scale-stable
        # signal regardless of how badly the reward shape blows up.
        self._ret_low = torch.tensor(0.0, device=device)
        self._ret_high = torch.tensor(0.0, device=device)

        # ── Adaptive Lagrangian for traversability term ─────────────────
        # log_lambda is stored unconstrained; we map via softplus so it's
        # always positive. Adapts each step by dual ascent on the
        # constraint "average traversability ≥ target".
        self._trav_lambda = torch.tensor(
            float(cfg.trav.init_lambda), device=device)

    # ──────────────────────────────────────────────────────────────────────
    def _move(self, batch):
        obs, actions, rewards, dones, _ = batch
        obs = {k: v.to(self.device) for k, v in obs.items()}
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        return obs, actions, rewards, dones

    # ──────────────────────────────────────────────────────────────────────
    # World-model update (encoders + RSSM + reward + continuation)
    # ──────────────────────────────────────────────────────────────────────
    def train_world_model(self, batch) -> Tuple[Dict[str, float], State]:
        cfg = self.cfg.train
        obs, actions, rewards, dones = self._move(batch)

        embeds = self.model.encode_obs_seq(obs)
        prior, post = self.model.rssm.observe(embeds, actions)
        feats = RSSM.get_feat(post)                  # [B, T, feat_dim]

        pred_reward = self.model.reward_head(feats).squeeze(-1)
        cont_logits = self.model.continuation_head.logits(feats)
        cont_target = 1.0 - dones

        # TODO(dreamerv3): swap MSE -> symlog two-hot regression on returns.
        loss_reward = F.mse_loss(pred_reward, rewards)
        loss_cont = F.binary_cross_entropy_with_logits(cont_logits, cont_target)
        loss_kl = self.model.rssm.kl_loss(
            prior, post, free_nats=cfg.free_nats, balance=cfg.kl_balance)
        # TODO(dreamerv3): observation reconstruction terms (BEV/IMU/goal).
        loss = loss_reward + loss_cont + loss_kl

        # ── Traversability head supervised loss ──────────────────────────
        trav_loss_val = 0.0
        if self.cfg.trav.enable:
            tilt = self._derive_tilt(obs)            # [B, T]
            lidar_max = self._derive_lidar_max(obs)  # [B, T]
            label = make_traversability_label(
                tilt, lidar_max,
                tilt_sigma=self.cfg.trav.label_tilt_sigma,
                lidar_sigma=self.cfg.trav.label_lidar_sigma,
            )
            pred_trav = self.model.traversability_head(feats)
            loss_trav = F.binary_cross_entropy(
                pred_trav.clamp(1e-5, 1 - 1e-5), label)
            loss = loss + self.cfg.trav.head_loss_weight * loss_trav
            trav_loss_val = float(loss_trav.detach())

        self.opt_world.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.world_model_parameters()), cfg.grad_clip)
        self.opt_world.step()

        last_state = {k: post[k][:, -1].detach() for k in post}
        metrics = {
            "loss/world_total": float(loss.detach()),
            "loss/reward": float(loss_reward.detach()),
            "loss/cont": float(loss_cont.detach()),
            "loss/kl": float(loss_kl.detach()),
        }
        if self.cfg.trav.enable:
            metrics["loss/trav"] = trav_loss_val
        return metrics, last_state

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _derive_tilt(obs):
        """Extract scalar tilt (roll/pitch hypot) from observation IMU vector.

        IMU layout per wrapper: ``imu = [linacc(3), angvel(3), rpy(3), bodyvel(3)]``,
        so roll = imu[..., 6], pitch = imu[..., 7].
        """
        imu = obs["imu"]
        roll = imu[..., 6]
        pitch = imu[..., 7]
        return torch.sqrt(roll * roll + pitch * pitch)

    @staticmethod
    def _derive_lidar_max(obs):
        """Max occupancy in the central window of the BEV — same proxy as
        the wrapper uses for the hand-written lidar penalty."""
        bev = obs["lidar_bev"]    # [B, T, 4, H, W]
        H = bev.shape[-1]
        r = max(1, H // 8)
        c = H // 2
        crop = bev[..., 0, c - r:c + r, c - r:c + r]    # occupancy channel
        return crop.reshape(*crop.shape[:-2], -1).amax(dim=-1)

    # ──────────────────────────────────────────────────────────────────────
    # Actor / critic via imagined rollouts
    # ──────────────────────────────────────────────────────────────────────
    def train_actor_critic(self, init_state: State) -> Dict[str, float]:
        cfg = self.cfg.train
        tcfg = self.cfg.trav
        states, _ = self.model.imagine(init_state, cfg.imagine_horizon)
        feats = RSSM.get_feat(states)                # [B, H, feat_dim]

        with torch.no_grad():
            # Use the *target* critic for value bootstrapping — bounds
            # estimation error inside the λ-return calculation.
            values = self._target_critic_net(feats).squeeze(-1)
            base_rewards = self.model.reward_head(feats).squeeze(-1)
            cont = self.model.continuation_head(feats).squeeze(-1)
            # ── Learned traversability shaping ─────────────────────────
            if tcfg.enable:
                trav = self.model.traversability_head(feats)   # [B, H]
                # Reward augmentation: (trav - 1) ∈ [-1, 0], so unsafe
                # latent regions soft-penalise the actor. Lambda is
                # adapted below.
                shaped_rewards = base_rewards + self._trav_lambda * (trav - 1.0)
                mean_trav = float(trav.mean().detach())
            else:
                shaped_rewards = base_rewards
                trav = None
                mean_trav = float("nan")

            returns_raw = self._lambda_return(
                shaped_rewards, values, cont, cfg.discount, cfg.lambda_gae)
            # ── Percentile return normalization (DreamerV3) ────────────
            if tcfg.enable_return_norm:
                p5 = torch.quantile(returns_raw, 0.05)
                p95 = torch.quantile(returns_raw, 0.95)
                d = tcfg.return_ema_decay
                self._ret_low = d * self._ret_low + (1 - d) * p5
                self._ret_high = d * self._ret_high + (1 - d) * p95
                scale = (self._ret_high - self._ret_low).clamp(
                    min=tcfg.return_norm_eps)
                returns = returns_raw / scale
                values_norm = values / scale
            else:
                returns = returns_raw
                values_norm = values

        # Actor: maximise (normalised) returns. Re-evaluate policy for grad flow.
        # In the demonstration-anchored regime the imagined-return loss is
        # down-weighted so BC dominates the actor update (option B in the
        # design discussion).
        actions, log_prob, _ = self.model.actor_critic.sample_action(feats)
        advantage = (returns - values_norm).detach()
        actor_loss_dreamer = (
            -(log_prob * advantage).mean()
            * self.cfg.bc.dreamer_actor_weight
        )

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss_dreamer.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            list(self.model.actor_parameters()), cfg.grad_clip)
        self.opt_actor.step()

        # Critic: regress predicted value onto raw returns (so the critic
        # still learns the true reward scale; only the actor sees the
        # normalised version).
        value_pred = self.model.actor_critic.critic(feats.detach())
        critic_loss = F.mse_loss(value_pred, returns_raw.detach())

        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.critic_parameters()), cfg.grad_clip)
        self.opt_critic.step()

        # ── EMA target-critic update ───────────────────────────────────────
        # θ_target ← ρ θ_target + (1-ρ) θ_online ; bounds value estimation.
        with torch.no_grad():
            rho = self._target_ema_decay
            for tp, op in zip(self._target_critic_net.parameters(),
                              self.model.actor_critic.critic_net.parameters()):
                tp.data.mul_(rho).add_(op.data, alpha=1 - rho)

        # ── Adaptive Lagrangian update for λ ─────────────────────────────
        # Gradient: dual ascent on  λ * (target - mean_trav). When mean
        # traversability is below target, λ grows; otherwise it shrinks.
        if tcfg.enable and trav is not None:
            with torch.no_grad():
                gap = tcfg.target_traversable_frac - trav.mean()
                self._trav_lambda = (
                    self._trav_lambda + tcfg.lambda_lr * gap
                ).clamp(min=tcfg.lambda_min, max=tcfg.lambda_max)

        metrics = {
            "loss/actor_dreamer": float(actor_loss_dreamer.detach()),
            "loss/critic": float(critic_loss.detach()),
            "stat/imag_return_mean": float(returns_raw.mean().detach()),
        }
        if tcfg.enable:
            metrics["stat/trav_mean"] = mean_trav
            metrics["stat/trav_lambda"] = float(self._trav_lambda)
        if tcfg.enable_return_norm:
            metrics["stat/ret_scale"] = float(
                (self._ret_high - self._ret_low).clamp(min=tcfg.return_norm_eps))
        return metrics

    # ──────────────────────────────────────────────────────────────────────
    # Behavior cloning (human demos) and recovery-BC
    # ──────────────────────────────────────────────────────────────────────
    def _bc_loss_from_batch(self, batch) -> torch.Tensor:
        obs, actions, _, _ = self._move(batch)
        embeds = self.model.encode_obs_seq(obs)
        _, post = self.model.rssm.observe(embeds, actions)
        feats = RSSM.get_feat(post)
        pred, _, _ = self.model.actor_critic.sample_action(feats, deterministic=True)
        return self.model.bc.compute_bc_loss(pred, actions)

    def _recovery_bc_loss_from_batch(self, batch) -> torch.Tensor:
        obs, actions, _, _ = self._move(batch)
        embeds = self.model.encode_obs_seq(obs)
        _, post = self.model.rssm.observe(embeds, actions)
        feats = RSSM.get_feat(post)
        pred, _, _ = self.model.actor_critic.sample_action(feats, deterministic=True)
        return self.model.bc.compute_recovery_bc_loss(pred, actions)

    def train_behavior_cloning(self, batch) -> Dict[str, float]:
        loss = self._bc_loss_from_batch(batch) * self.cfg.bc.behavior_cloning_weight
        self.opt_actor.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.actor_parameters()), self.cfg.train.grad_clip)
        self.opt_actor.step()
        return {"loss/bc": float(loss.detach())}

    def train_recovery_behavior_cloning(self, batch) -> Dict[str, float]:
        loss = self._recovery_bc_loss_from_batch(batch) \
            * self.cfg.bc.recovery_behavior_cloning_weight
        self.opt_actor.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.actor_parameters()), self.cfg.train.grad_clip)
        self.opt_actor.step()
        return {"loss/recovery_bc": float(loss.detach())}

    # ──────────────────────────────────────────────────────────────────────
    # One full training step
    # ──────────────────────────────────────────────────────────────────────
    def train_step(self, buffer: ReplayBuffer) -> Optional[Dict[str, float]]:
        cfg = self.cfg.train
        if not buffer.can_sample(cfg.batch_size, cfg.seq_len):
            return None

        # 1) World model on a normal sequence batch.
        batch = buffer.sample_sequence_batch(cfg.batch_size, cfg.seq_len)
        world_metrics, last_state = self.train_world_model(batch)

        # 2) Actor-critic from imagined rollouts.
        ac_metrics = self.train_actor_critic(last_state)

        metrics: Dict[str, float] = {**world_metrics, **ac_metrics}

        # 3) Human-demo BC (if any exists).
        try:
            human_batch = buffer.sample_human_demo_batch(cfg.batch_size, cfg.seq_len)
        except Exception:
            human_batch = None
        if human_batch is not None:
            metrics.update(self.train_behavior_cloning(human_batch))

        # 4) Recovery-BC (if any recovery demos exist).
        if self.cfg.failed.enable_human_recovery_learning:
            try:
                recov_batch = buffer.sample_recovery_demo_batch(cfg.batch_size, cfg.seq_len)
            except Exception:
                recov_batch = None
            if recov_batch is not None:
                metrics.update(self.train_recovery_behavior_cloning(recov_batch))

        self._last_metrics = metrics
        return metrics

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _lambda_return(
        rewards: torch.Tensor,    # [B, H]
        values: torch.Tensor,     # [B, H]
        cont: torch.Tensor,       # [B, H]
        discount: float,
        lam: float,
    ) -> torch.Tensor:
        B, H = rewards.shape
        returns = torch.zeros_like(rewards)
        next_return = values[:, -1]
        for t in reversed(range(H)):
            bootstrap = (1 - lam) * values[:, t] + lam * next_return
            returns[:, t] = rewards[:, t] + discount * cont[:, t] * bootstrap
            next_return = returns[:, t]
        return returns

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoints
    # ──────────────────────────────────────────────────────────────────────
    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "opt_world": self.opt_world.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "last_metrics": self._last_metrics,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt_world.load_state_dict(ckpt["opt_world"])
        self.opt_actor.load_state_dict(ckpt["opt_actor"])
        self.opt_critic.load_state_dict(ckpt["opt_critic"])
