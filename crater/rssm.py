"""Recurrent State-Space Model (Dreamer-style).

State is a dict with keys (deter, stoch, mean, std):
    deter : [B, H]  deterministic GRU hidden
    stoch : [B, S]  stochastic latent sample
    mean  : [B, S]  distribution mean (used for KL)
    std   : [B, S]  distribution stddev (used for KL)

Distributions are continuous diagonal Gaussians for this scaffold. Swap the
`_dist` head + sampling path for categorical when moving to full DreamerV3.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


State = Dict[str, torch.Tensor]


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

        # Pre-GRU combines previous stochastic state and action.
        self.pre_gru = nn.Sequential(
            nn.Linear(stoch + action_dim, hidden),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden, hidden)

        # Prior head: predicts z distribution from deter only.
        self.prior_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * stoch),
        )
        # Posterior head: predicts z from (deter, obs_embed).
        self.post_head = nn.Sequential(
            nn.Linear(hidden + obs_embed, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * stoch),
        )

    # ──────────────────────────────────────────────────────────────────────
    def _dist(self, params: torch.Tensor) -> Normal:
        mean, std = params.chunk(2, dim=-1)
        std = F.softplus(std) + self.min_std
        return Normal(mean, std)

    def init_state(self, batch_size: int, device: torch.device) -> State:
        return {
            "deter": torch.zeros(batch_size, self.hidden, device=device),
            "stoch": torch.zeros(batch_size, self.stoch, device=device),
            "mean":  torch.zeros(batch_size, self.stoch, device=device),
            "std":   torch.ones(batch_size, self.stoch, device=device),
        }

    # ──────────────────────────────────────────────────────────────────────
    def imagine_step(self, prev: State, action: torch.Tensor) -> State:
        """Roll the GRU one step using the prior — no observation needed."""
        x = self.pre_gru(torch.cat([prev["stoch"], action], dim=-1))
        deter = self.gru(x, prev["deter"])
        prior = self._dist(self.prior_head(deter))
        stoch = prior.rsample()
        return {"deter": deter, "stoch": stoch,
                "mean": prior.mean, "std": prior.stddev}

    def observe_step(
        self,
        prev: State,
        prev_action: torch.Tensor,
        obs_embed: torch.Tensor,
    ) -> Tuple[State, State]:
        """Single posterior step. Returns (prior_state, posterior_state)."""
        x = self.pre_gru(torch.cat([prev["stoch"], prev_action], dim=-1))
        deter = self.gru(x, prev["deter"])
        prior_params = self.prior_head(deter)
        post_params = self.post_head(torch.cat([deter, obs_embed], dim=-1))
        prior = self._dist(prior_params)
        post = self._dist(post_params)
        prior_state = {"deter": deter, "stoch": prior.rsample(),
                       "mean": prior.mean, "std": prior.stddev}
        post_state = {"deter": deter, "stoch": post.rsample(),
                      "mean": post.mean, "std": post.stddev}
        return prior_state, post_state

    # ──────────────────────────────────────────────────────────────────────
    def observe(
        self,
        embeds: torch.Tensor,    # [B, T, obs_embed]
        actions: torch.Tensor,   # [B, T, A]
        init: State | None = None,
    ) -> Tuple[State, State]:
        """Posterior rollout. Each field of returned dicts has shape [B, T, ...]."""
        B, T, _ = embeds.shape
        if init is None:
            init = self.init_state(B, embeds.device)

        prior_acc: Dict[str, list] = {k: [] for k in ("deter", "stoch", "mean", "std")}
        post_acc: Dict[str, list] = {k: [] for k in ("deter", "stoch", "mean", "std")}
        prev = init
        for t in range(T):
            prior, post = self.observe_step(prev, actions[:, t], embeds[:, t])
            for k in prior_acc:
                prior_acc[k].append(prior[k])
                post_acc[k].append(post[k])
            prev = post
        stack = lambda xs: torch.stack(xs, dim=1)
        return {k: stack(v) for k, v in prior_acc.items()}, \
               {k: stack(v) for k, v in post_acc.items()}

    def imagine(
        self,
        init: State,
        policy_fn,                     # callable(feat) -> action
        horizon: int,
    ) -> Tuple[State, torch.Tensor]:
        """Imagine `horizon` steps. Stacked [B, H, ...] state + actions."""
        acc: Dict[str, list] = {k: [] for k in ("deter", "stoch", "mean", "std")}
        actions = []
        prev = init
        for _ in range(horizon):
            feat = self.get_feat(prev)
            action = policy_fn(feat)
            prev = self.imagine_step(prev, action)
            for k in acc:
                acc[k].append(prev[k])
            actions.append(action)
        stack = lambda xs: torch.stack(xs, dim=1)
        out = {k: stack(v) for k, v in acc.items()}
        return out, torch.stack(actions, dim=1)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_feat(state: State) -> torch.Tensor:
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    @staticmethod
    def kl_loss(
        prior: State, post: State,
        free_nats: float = 1.0, balance: float = 0.8,
    ) -> torch.Tensor:
        """DreamerV3-style KL balancing."""
        prior_d = Normal(prior["mean"], prior["std"])
        post_d = Normal(post["mean"], post["std"])
        prior_d_stop = Normal(prior["mean"].detach(), prior["std"].detach())
        post_d_stop = Normal(post["mean"].detach(), post["std"].detach())
        kl_lhs = kl_divergence(post_d_stop, prior_d).sum(-1)
        kl_rhs = kl_divergence(post_d, prior_d_stop).sum(-1)
        kl = balance * kl_lhs + (1.0 - balance) * kl_rhs
        return torch.clamp(kl, min=free_nats).mean()
