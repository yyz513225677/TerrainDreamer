"""
Recurrent State Space Model (RSSM)
====================================
The core dynamics model, inspired by DreamerV3.

State = (deterministic h, stochastic z)
  - h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})     [deterministic transition]
  - z_t ~ p(z_t | h_t)                         [prior — imagination]
  - z_t ~ q(z_t | h_t, o_t)                    [posterior — with observation]

This allows "dreaming":
  Given h_t and action sequence, rollout future states without observations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class RSSMState:
    """Combined deterministic + stochastic state."""
    deter: torch.Tensor    # [B, deter_dim] — GRU hidden state
    stoch: torch.Tensor    # [B, stoch_dim * stoch_classes] — categorical latent
    logits: torch.Tensor   # [B, stoch_dim, stoch_classes] — distribution params

    @property
    def feature(self) -> torch.Tensor:
        """Concatenated state for downstream heads."""
        return torch.cat([self.deter, self.stoch], dim=-1)

    def detach(self) -> "RSSMState":
        return RSSMState(
            deter=self.deter.detach(),
            stoch=self.stoch.detach(),
            logits=self.logits.detach(),
        )


class RSSM(nn.Module):
    """
    Recurrent State Space Model with discrete latent variables.

    The model learns to predict how terrain states evolve under actions,
    enabling "imagination" — rolling out future states without real sensor data.
    """

    def __init__(
        self,
        embed_dim: int = 256,     # Terrain encoder output
        action_dim: int = 2,      # [steering, throttle]
        deter_dim: int = 512,     # GRU hidden size
        stoch_dim: int = 64,      # Number of categorical variables
        stoch_classes: int = 64,  # Classes per variable
        hidden_dim: int = 512,    # MLP hidden size
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.stoch_total = stoch_dim * stoch_classes

        # ---- Sequence model (deterministic path) ----
        # Input to GRU: previous stochastic state + action
        self.pre_gru = nn.Sequential(
            nn.Linear(self.stoch_total + action_dim, hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

        # ---- Prior (imagination): p(z_t | h_t) ----
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_total),
        )

        # ---- Posterior (observation): q(z_t | h_t, o_t) ----
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_total),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """Create zero-initialized RSSM state."""
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.stoch_total, device=device)
        logits = torch.zeros(
            batch_size, self.stoch_dim, self.stoch_classes, device=device
        )
        return RSSMState(deter=deter, stoch=stoch, logits=logits)

    def observe_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,        # [B, action_dim]
        observation: torch.Tensor,   # [B, embed_dim] from terrain encoder
    ) -> Tuple[RSSMState, RSSMState]:
        """
        One step with an actual observation (training / online).

        Returns:
            prior_state: p(z_t | h_t) — what the model predicted
            posterior_state: q(z_t | h_t, o_t) — corrected with observation
        """
        # Deterministic transition
        deter = self._deterministic_step(prev_state, action)

        # Prior (model's prediction without seeing observation)
        prior_logits = self.prior_net(deter)
        prior_logits = prior_logits.reshape(-1, self.stoch_dim, self.stoch_classes)
        prior_stoch = self._sample_stochastic(prior_logits)
        prior_state = RSSMState(deter=deter, stoch=prior_stoch, logits=prior_logits)

        # Posterior (corrected with observation)
        post_input = torch.cat([deter, observation], dim=-1)
        post_logits = self.posterior_net(post_input)
        post_logits = post_logits.reshape(-1, self.stoch_dim, self.stoch_classes)
        post_stoch = self._sample_stochastic(post_logits)
        post_state = RSSMState(deter=deter, stoch=post_stoch, logits=post_logits)

        return prior_state, post_state

    def imagine_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
    ) -> RSSMState:
        """
        One step WITHOUT observation (dreaming / planning).
        Uses prior only.
        """
        deter = self._deterministic_step(prev_state, action)
        prior_logits = self.prior_net(deter)
        prior_logits = prior_logits.reshape(-1, self.stoch_dim, self.stoch_classes)
        prior_stoch = self._sample_stochastic(prior_logits)
        return RSSMState(deter=deter, stoch=prior_stoch, logits=prior_logits)

    def observe_sequence(
        self,
        observations: torch.Tensor,  # [B, T, embed_dim]
        actions: torch.Tensor,        # [B, T, action_dim]
        initial_state: Optional[RSSMState] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Process a full sequence with observations (for training).

        Returns dicts with keys: deter, stoch, logits — each [B, T, ...]
        """
        B, T, _ = observations.shape
        device = observations.device

        if initial_state is None:
            state = self.initial_state(B, device)
        else:
            state = initial_state

        priors = {"deter": [], "stoch": [], "logits": []}
        posteriors = {"deter": [], "stoch": [], "logits": []}

        for t in range(T):
            prior, posterior = self.observe_step(
                state, actions[:, t], observations[:, t]
            )
            # Use posterior as next state (teacher forcing)
            state = posterior

            for key, val in [
                ("deter", prior.deter), ("stoch", prior.stoch),
                ("logits", prior.logits)
            ]:
                priors[key].append(val)
            for key, val in [
                ("deter", posterior.deter), ("stoch", posterior.stoch),
                ("logits", posterior.logits)
            ]:
                posteriors[key].append(val)

        # Stack: list of [B, ...] → [B, T, ...]
        priors = {k: torch.stack(v, dim=1) for k, v in priors.items()}
        posteriors = {k: torch.stack(v, dim=1) for k, v in posteriors.items()}

        return priors, posteriors

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _deterministic_step(
        self, prev_state: RSSMState, action: torch.Tensor
    ) -> torch.Tensor:
        """GRU transition: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})."""
        gru_input = torch.cat([prev_state.stoch, action], dim=-1)
        gru_input = self.pre_gru(gru_input)
        deter = self.gru(gru_input, prev_state.deter)
        return deter

    def _sample_stochastic(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample from categorical distribution with straight-through gradient.
        Uses uniform prior mixture (DreamerV3) to prevent posterior collapse.
        logits: [B, stoch_dim, stoch_classes]
        returns: [B, stoch_dim * stoch_classes] one-hot flattened
        """
        # Mix with uniform distribution to prevent collapse (DreamerV3 Section 3)
        uniform = torch.ones_like(logits) / self.stoch_classes
        probs = F.softmax(logits, dim=-1)
        mixed_probs = 0.99 * probs + 0.01 * uniform

        dist = torch.distributions.OneHotCategorical(probs=mixed_probs)
        sample = dist.sample()  # [B, stoch_dim, stoch_classes] one-hot

        # Straight-through: gradient flows through mixed_probs
        sample = sample + mixed_probs - mixed_probs.detach()

        return sample.reshape(sample.shape[0], -1)

    @staticmethod
    def kl_loss(
        prior_logits: torch.Tensor,
        posterior_logits: torch.Tensor,
        balance: float = 0.8,
        free_nats: float = 1.0,
    ) -> torch.Tensor:
        """
        KL divergence between posterior and prior (categorical).
        Uses KL balancing (DreamerV3): blend of forward and reverse KL.
        """
        prior_dist = torch.distributions.OneHotCategorical(logits=prior_logits)
        post_dist = torch.distributions.OneHotCategorical(logits=posterior_logits)

        # Forward KL: push prior toward posterior
        kl_forward = torch.distributions.kl_divergence(
            post_dist, prior_dist
        ).sum(-1)  # Sum over stoch_dim

        # Reverse KL: push posterior toward prior
        kl_reverse = torch.distributions.kl_divergence(
            prior_dist, post_dist
        ).sum(-1)

        # Balance
        kl = balance * kl_forward + (1 - balance) * kl_reverse

        # Free nats
        kl = torch.clamp(kl, min=free_nats)

        return kl.mean()
