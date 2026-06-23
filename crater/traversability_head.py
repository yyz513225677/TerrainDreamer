"""TraversabilityHead — a learned local cost-map distilled into the world
model's latent.

Given an RSSM feature ``[B, ..., feat_dim]`` it predicts a scalar in
``[0, 1]`` interpreted as "probability that the rover can traverse this
state".

During training we generate pseudo-ground-truth labels from the observed
``tilt`` and BEV ``lidar_occupancy_max`` signals — both already available
from the wrapper. The head then runs inside imagined rollouts so the
actor-critic gets a *learned, smooth* traversability gradient instead of
the hand-written ``lidar_penalty`` / ``tilt_penalty`` we used before.

Novelty over standard Dreamer:
  - World model gets an extra head that learns to predict traversability
    from latent state alone (no perception module call at imagination
    time).
  - Reward is augmented inside imagination by ``λ * (head(feat) - 1)``
    so unsafe latent regions are softly avoided.
  - The weight ``λ`` is adapted by a Lagrangian-style dual variable that
    targets a configurable "fraction of traversable states" in the
    imagined rollouts. This decouples the user from manually tuning a
    penalty weight that wrecks critic scale (the issue we hit earlier
    when ``imag_return_mean`` locked into −500).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TraversabilityHead(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        layers = []
        d = feat_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Return P(traversable) ∈ [0, 1], shape = feat shape minus last dim."""
        return torch.sigmoid(self.net(feat).squeeze(-1))


def make_traversability_label(
    tilt: torch.Tensor,
    lidar_max: torch.Tensor,
    tilt_sigma: float = 0.35,
    lidar_sigma: float = 0.35,
) -> torch.Tensor:
    """Build a pseudo-ground-truth label for the traversability head.

    Both signals decay as Gaussian away from ``0``. A near-flat rover with
    no nearby obstacles → label ≈ 1; a tipped-over rover next to a wall →
    label ≈ 0. The final label is the product of both decays, which
    matches the multiplicative interpretation in the safety shield.

    Args:
        tilt:       [B, ...] absolute roll-pitch magnitude in rad.
        lidar_max:  [B, ...] max BEV-occupancy near the rover in [0, 1].
    """
    a = torch.exp(-(tilt / tilt_sigma) ** 2)
    b = torch.exp(-(lidar_max / lidar_sigma) ** 2)
    return (a * b).clamp(0.0, 1.0)
