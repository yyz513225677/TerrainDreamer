"""
TerrainDreamer World Model
============================
Complete world model that wraps:
  - Terrain Encoder (point cloud → latent)
  - RSSM (latent dynamics)
  - Decoder heads (latent → predictions)

Decoder heads:
  1. Observation decoder  — reconstruct terrain features
  2. Reward predictor     — predict traversability reward
  3. Continue predictor   — predict episode continuation (not stuck/flipped)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from terrain_dreamer.preprocessing.terrain_encoder import TerrainEncoder
from terrain_dreamer.world_model.rssm import RSSM, RSSMState


class ObservationDecoder(nn.Module):
    """Decode latent state back to terrain feature statistics."""

    def __init__(self, state_dim: int, output_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class RewardPredictor(nn.Module):
    """
    Predict scalar reward from latent state.
    Reward encodes: goal proximity + terrain traversability.
    Uses symlog binning (DreamerV3 style) for stable training.
    """

    def __init__(self, state_dim: int, hidden: int = 512, num_bins: int = 255):
        super().__init__()
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_bins),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns reward logits [B, num_bins]."""
        return self.net(features)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Returns scalar reward estimate [B]."""
        logits = self.forward(features)
        probs = F.softmax(logits, dim=-1)
        # Bin centers: symlog-spaced
        bins = torch.linspace(-20, 20, self.num_bins, device=features.device)
        return (probs * bins).sum(dim=-1)


class ContinuePredictor(nn.Module):
    """Predict probability of episode continuing (not terminated)."""

    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns continue logit [B, 1]."""
        return self.net(features)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Returns continue probability [B]."""
        return torch.sigmoid(self.forward(features)).squeeze(-1)


class TerrainDreamerModel(nn.Module):
    """
    Full world model for off-road autonomous driving.

    Pipeline:
      Observation → Encoder → RSSM (dynamics) → Decoder heads
                                ↕
                    Imagination (dreaming future states)
    """

    def __init__(
        self,
        # Encoder
        input_channels: int = 8,
        embed_dim: int = 256,
        sa_npoints: List[int] = None,
        sa_nsamples: List[int] = None,
        sa_radii: List[float] = None,
        # RSSM
        action_dim: int = 2,
        deter_dim: int = 512,
        stoch_dim: int = 64,
        stoch_classes: int = 64,
        hidden_dim: int = 512,
        # Loss weights
        kl_weight: float = 1.0,
        kl_balance: float = 0.8,
        free_nats: float = 1.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.kl_weight = kl_weight
        self.kl_balance = kl_balance
        self.free_nats = free_nats

        # State dimension (deterministic + stochastic)
        state_dim = deter_dim + stoch_dim * stoch_classes

        encoder_kwargs = dict(input_channels=input_channels, feature_dim=embed_dim)
        if sa_npoints is not None:
            encoder_kwargs["sa_npoints"] = sa_npoints
        if sa_nsamples is not None:
            encoder_kwargs["sa_nsamples"] = sa_nsamples
        if sa_radii is not None:
            encoder_kwargs["sa_radii"] = sa_radii

        # Components
        self.encoder = TerrainEncoder(**encoder_kwargs)

        self.rssm = RSSM(
            embed_dim=embed_dim,
            action_dim=action_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            stoch_classes=stoch_classes,
            hidden_dim=hidden_dim,
        )

        self.obs_decoder = ObservationDecoder(state_dim, embed_dim)
        self.reward_pred = RewardPredictor(state_dim)
        self.continue_pred = ContinuePredictor(state_dim)

        # Auxiliary: stochastic-only decoder to prevent posterior collapse
        stoch_total = stoch_dim * stoch_classes
        self.stoch_decoder = ObservationDecoder(stoch_total, embed_dim, hidden=256)

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        """Encode point cloud to latent embedding. [B, N, C] → [B, embed_dim]."""
        return self.encoder(points)

    def observe(
        self,
        observations: torch.Tensor,  # [B, T, embed_dim] pre-encoded
        actions: torch.Tensor,        # [B, T, action_dim]
        initial_state: Optional[RSSMState] = None,
    ) -> Tuple[Dict, Dict]:
        """
        Process sequence with observations (for training).
        Returns prior and posterior state dicts.
        """
        return self.rssm.observe_sequence(observations, actions, initial_state)

    def imagine(
        self,
        initial_state: RSSMState,
        action_sequence: torch.Tensor,  # [B, H, action_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Dream forward H steps from initial state using action sequence.
        No observations needed — pure imagination.
        """
        B, H, _ = action_sequence.shape
        state = initial_state

        states = {"deter": [], "stoch": [], "features": []}

        for t in range(H):
            state = self.rssm.imagine_step(state, action_sequence[:, t])
            states["deter"].append(state.deter)
            states["stoch"].append(state.stoch)
            states["features"].append(state.feature)

        states = {k: torch.stack(v, dim=1) for k, v in states.items()}
        return states

    def training_loss(
        self,
        points_seq: torch.Tensor,   # [B, T, N, C] raw point clouds
        actions: torch.Tensor,        # [B, T, action_dim]
        rewards: torch.Tensor,        # [B, T]
        continues: torch.Tensor,      # [B, T] (1=continue, 0=terminal)
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all world model losses for a training batch.

        Returns dict of individual losses + total loss.
        """
        B, T = actions.shape[:2]

        # Encode each timestep
        # Reshape: [B*T, N, C]
        BT = B * T
        N, C = points_seq.shape[2], points_seq.shape[3]
        flat_points = points_seq.reshape(BT, N, C)
        flat_embeds = self.encoder(flat_points)  # [B*T, embed_dim]
        observations = flat_embeds.reshape(B, T, -1)  # [B, T, embed_dim]

        # Run RSSM
        priors, posteriors = self.rssm.observe_sequence(observations, actions)

        # Posterior features for decoder heads
        post_features = torch.cat(
            [posteriors["deter"], posteriors["stoch"]], dim=-1
        )  # [B, T, state_dim]

        # --- Losses ---

        # 1. KL divergence (prior ↔ posterior)
        kl_loss = RSSM.kl_loss(
            priors["logits"], posteriors["logits"],
            balance=self.kl_balance, free_nats=self.free_nats,
        )

        # 2. Observation reconstruction
        obs_pred = self.obs_decoder(post_features)  # [B, T, embed_dim]
        obs_loss = F.mse_loss(obs_pred, observations.detach())

        # 3. Reward prediction
        reward_logits = self.reward_pred(post_features)  # [B, T, num_bins]
        # Simple MSE on scalar prediction for now
        reward_pred = self.reward_pred.predict(
            post_features.reshape(-1, post_features.shape[-1])
        ).reshape(B, T)
        reward_loss = F.mse_loss(reward_pred, rewards)

        # 4. Continue prediction
        continue_logits = self.continue_pred(post_features)  # [B, T, 1]
        continue_loss = F.binary_cross_entropy_with_logits(
            continue_logits.squeeze(-1), continues
        )

        # 5. Stochastic-only reconstruction (prevents posterior collapse)
        stoch_pred = self.stoch_decoder(posteriors["stoch"])  # [B, T, embed_dim]
        stoch_loss = F.mse_loss(stoch_pred, observations.detach())

        # Total
        total_loss = (
            obs_loss
            + self.kl_weight * kl_loss
            + reward_loss
            + continue_loss
            + 1.0 * stoch_loss
        )

        return {
            "total": total_loss,
            "kl": kl_loss,
            "obs_recon": obs_loss,
            "reward": reward_loss,
            "continue": continue_loss,
            "stoch_recon": stoch_loss,
        }
