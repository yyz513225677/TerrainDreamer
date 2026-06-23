"""Recurrent State-Space Model (Dreamer-style).

State = (h, z) where
    h : deterministic GRU hidden, [B, rssm_hidden]
    z : stochastic latent sample, [B, rssm_stoch]

Distributions over `z` are continuous diagonal Gaussians.
For DreamerV3 categorical latents, swap `_dist` and the sampling pathway.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


@dataclass
class RSSMState:
    h: torch.Tensor          # [B, H]
    z: torch.Tensor          # [B, S]
    mean: torch.Tensor       # [B, S]
    std: torch.Tensor        # [B, S]

    def detach(self) -> "RSSMState":
        return RSSMState(self.h.detach(), self.z.detach(),
                         self.mean.detach(), self.std.detach())


class RSSM(nn.Module):
    def __init__(
        self,
        action_dim: int = 2,
        obs_embed: int = 768,
        hidden: int = 1024,
        stoch: int = 32,
        min_std: float = 0.1,
    ):
        super().__init__()
        self.hidden = hidden
        self.stoch = stoch
        self.min_std = min_std

        # Pre-GRU: combine previous stochastic state and action.
        self.pre_gru = nn.Sequential(
            nn.Linear(stoch + action_dim, hidden),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden, hidden)

        # Prior head: predicts z distribution from h only.
        self.prior_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * stoch),
        )
        # Posterior head: predicts z from (h, obs_embed).
        self.post_head = nn.Sequential(
            nn.Linear(hidden + obs_embed, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * stoch),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Distribution helpers
    # ──────────────────────────────────────────────────────────────────────
    def _dist(self, params: torch.Tensor) -> Normal:
        mean, std = params.chunk(2, dim=-1)
        std = F.softplus(std) + self.min_std
        return Normal(mean, std)

    def initial_state(self, batch: int, device: torch.device) -> RSSMState:
        h = torch.zeros(batch, self.hidden, device=device)
        z = torch.zeros(batch, self.stoch, device=device)
        mean = torch.zeros_like(z)
        std = torch.ones_like(z)
        return RSSMState(h, z, mean, std)

    # ──────────────────────────────────────────────────────────────────────
    # Single-step transitions
    # ──────────────────────────────────────────────────────────────────────
    def imagine_step(self, prev: RSSMState, prev_action: torch.Tensor) -> RSSMState:
        """Roll the GRU one step using the prior — no observation needed."""
        x = self.pre_gru(torch.cat([prev.z, prev_action], dim=-1))
        h = self.gru(x, prev.h)
        prior = self._dist(self.prior_head(h))
        z = prior.rsample()
        return RSSMState(h, z, prior.mean, prior.stddev)

    def observe_step(
        self,
        prev: RSSMState,
        prev_action: torch.Tensor,
        obs_embed: torch.Tensor,
    ) -> Tuple[RSSMState, RSSMState]:
        """Single posterior step. Returns (prior, posterior) so the trainer
        can compute the KL between them."""
        x = self.pre_gru(torch.cat([prev.z, prev_action], dim=-1))
        h = self.gru(x, prev.h)
        prior_params = self.prior_head(h)
        post_params = self.post_head(torch.cat([h, obs_embed], dim=-1))
        prior = self._dist(prior_params)
        post = self._dist(post_params)
        z = post.rsample()
        post_state = RSSMState(h, z, post.mean, post.stddev)
        prior_state = RSSMState(h, prior.rsample(), prior.mean, prior.stddev)
        return prior_state, post_state

    # ──────────────────────────────────────────────────────────────────────
    # Multi-step rollouts
    # ──────────────────────────────────────────────────────────────────────
    def observe(
        self,
        embeds: torch.Tensor,    # [B, T, obs_embed]
        actions: torch.Tensor,   # [B, T, A]
        init: RSSMState | None = None,
    ) -> Tuple[RSSMState, RSSMState]:
        """Posterior rollout. Returns stacked (prior, posterior) over T steps;
        each field has shape [B, T, ...]."""
        B, T, _ = embeds.shape
        if init is None:
            init = self.initial_state(B, embeds.device)
        priors_h, priors_z, priors_m, priors_s = [], [], [], []
        posts_h, posts_z, posts_m, posts_s = [], [], [], []
        prev = init
        for t in range(T):
            prior, post = self.observe_step(prev, actions[:, t], embeds[:, t])
            priors_h.append(prior.h); priors_z.append(prior.z)
            priors_m.append(prior.mean); priors_s.append(prior.std)
            posts_h.append(post.h); posts_z.append(post.z)
            posts_m.append(post.mean); posts_s.append(post.std)
            prev = post
        stack = lambda xs: torch.stack(xs, dim=1)
        prior = RSSMState(stack(priors_h), stack(priors_z),
                          stack(priors_m), stack(priors_s))
        post = RSSMState(stack(posts_h), stack(posts_z),
                         stack(posts_m), stack(posts_s))
        return prior, post

    def imagine(
        self,
        init: RSSMState,
        policy_fn,                     # callable(feat) -> action
        horizon: int,
    ) -> Tuple[RSSMState, torch.Tensor]:
        """Imagine `horizon` steps. Returns stacked states [B, H, ...] and
        the actions sampled by `policy_fn` along the way."""
        states_h, states_z, states_m, states_s = [], [], [], []
        actions = []
        prev = init
        for _ in range(horizon):
            feat = self.get_feat(prev)
            action = policy_fn(feat)
            prev = self.imagine_step(prev, action)
            states_h.append(prev.h); states_z.append(prev.z)
            states_m.append(prev.mean); states_s.append(prev.std)
            actions.append(action)
        stack = lambda xs: torch.stack(xs, dim=1)
        out = RSSMState(stack(states_h), stack(states_z),
                        stack(states_m), stack(states_s))
        return out, torch.stack(actions, dim=1)

    # ──────────────────────────────────────────────────────────────────────
    # Features
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_feat(state: RSSMState) -> torch.Tensor:
        return torch.cat([state.h, state.z], dim=-1)

    @staticmethod
    def kl_loss(prior: RSSMState, post: RSSMState,
                free_nats: float = 1.0, balance: float = 0.8) -> torch.Tensor:
        """DreamerV3-style KL balance: blend KL(stop_grad(post) || prior)
        with KL(post || stop_grad(prior))."""
        prior_d = Normal(prior.mean, prior.std)
        post_d = Normal(post.mean, post.std)
        prior_d_stop = Normal(prior.mean.detach(), prior.std.detach())
        post_d_stop = Normal(post.mean.detach(), post.std.detach())
        kl_lhs = kl_divergence(post_d_stop, prior_d).sum(-1)
        kl_rhs = kl_divergence(post_d, prior_d_stop).sum(-1)
        kl = balance * kl_lhs + (1.0 - balance) * kl_rhs
        return torch.clamp(kl, min=free_nats).mean()
