"""
DreamerPolicy — Actor-Critic on top of the RSSM latent state
=============================================================
Implements:
  DreamerActor  : RSSM_state + goal_obs → action
  DreamerCritic : RSSM_state + goal_obs → value
  imagine_train  : imagination-based behavior learning (DreamerV3 style)

The actor learns to maximise imagined λ-returns predicted by the world model.
The critic learns to fit those returns as bootstrap targets.

State dim = deter_dim + stoch_dim * stoch_classes   (matches TerrainDreamerModel)
Goal dim  = 4   (obs[373:377]: dx_norm, dy_norm, dist_norm, heading_err_norm)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from terrain_dreamer.world_model.rssm import RSSMState


# ──────────────────────────────────────────────────────────────────────────────
# Actor
# ──────────────────────────────────────────────────────────────────────────────

class DreamerActor(nn.Module):
    """
    Gaussian actor that maps (RSSM state, goal) → (lin_vel, ang_vel).

    Outputs mean + log_std. During collection a small exploration noise
    is added. At evaluation, the mean is used directly.
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int = 4,
        action_dim: int = 2,
        hidden: int = 256,
        init_std: float = 0.5,
        min_std: float = 0.05,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.min_std    = min_std

        inp = state_dim + goal_dim
        self.trunk = nn.Sequential(
            nn.Linear(inp, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
        )
        self.mean_head   = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

        # Init output weights small so early actions are near zero
        nn.init.uniform_(self.mean_head.weight,   -0.01, 0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.constant_(self.log_std_head.bias, float(init_std))

    def forward(
        self,
        state_feat: torch.Tensor,  # [B, state_dim]
        goal_obs:   torch.Tensor,  # [B, goal_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, std) each [B, action_dim]."""
        x    = self.trunk(torch.cat([state_feat, goal_obs], dim=-1))
        mean = torch.tanh(self.mean_head(x))
        std  = F.softplus(self.log_std_head(x)) + self.min_std
        return mean, std

    @torch.no_grad()
    def act(
        self,
        state_feat: torch.Tensor,
        goal_obs:   torch.Tensor,
        explore: bool = True,
        explore_scale: float = 1.0,
    ) -> torch.Tensor:
        """Sample (explore) or take mean (eval). Returns [B, action_dim]."""
        mean, std = self(state_feat, goal_obs)
        if explore:
            noise  = torch.randn_like(mean)
            action = torch.clamp(mean + explore_scale * std * noise, -1.0, 1.0)
        else:
            action = mean
        return action


# ──────────────────────────────────────────────────────────────────────────────
# Critic
# ──────────────────────────────────────────────────────────────────────────────

class DreamerCritic(nn.Module):
    """
    Value network:  (RSSM state, goal) → scalar V(s, g).
    Also keeps a slow EMA target network for stable training.
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int = 4,
        hidden: int = 256,
        ema_decay: float = 0.98,
    ):
        super().__init__()
        inp = state_dim + goal_dim
        self.net = nn.Sequential(
            nn.Linear(inp, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )
        # Slow-moving target (no gradient)
        self._target = nn.Sequential(
            nn.Linear(inp, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )
        self._ema_decay = ema_decay
        self._update_target(tau=0.0)    # copy weights at init

    def forward(
        self,
        state_feat: torch.Tensor,
        goal_obs:   torch.Tensor,
    ) -> torch.Tensor:
        """Returns value [B] from the online network."""
        x = torch.cat([state_feat, goal_obs], dim=-1)
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def target(
        self,
        state_feat: torch.Tensor,
        goal_obs:   torch.Tensor,
    ) -> torch.Tensor:
        """Bootstrap value from the slow EMA target. No gradient."""
        x = torch.cat([state_feat, goal_obs], dim=-1)
        return self._target(x).squeeze(-1)

    def update_target(self):
        """EMA update: target ← decay·target + (1-decay)·online."""
        self._update_target(tau=self._ema_decay)

    def _update_target(self, tau: float):
        for p_o, p_t in zip(self.net.parameters(), self._target.parameters()):
            p_t.data.mul_(tau).add_(p_o.data * (1.0 - tau))


# ──────────────────────────────────────────────────────────────────────────────
# Imagination trainer
# ──────────────────────────────────────────────────────────────────────────────

def imagine_train(
    model,                        # TerrainDreamerModel
    actor:   DreamerActor,
    critic:  DreamerCritic,
    actor_opt:  torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    start_states: Dict[str, torch.Tensor],  # posterior state tensors [B,T,...]
    start_goals:  torch.Tensor,             # [B,T,goal_dim]
    device: torch.device,
    H: int    = 15,
    gamma: float = 0.99,
    lam:   float = 0.95,
    entropy_scale: float = 3e-4,
    grad_clip: float = 100.0,
) -> Dict[str, float]:
    """
    Imagination-based behavior learning (DreamerV3 Section 2.4).

    1. Flatten start states from the replay buffer [B*T, state_dim].
    2. Roll out H imagination steps using the actor (no real observations).
    3. Compute rewards + continues with world model heads.
    4. Compute λ-returns as critic training targets.
    5. Update actor (maximise returns) and critic (fit targets).

    Returns dict of scalar losses.
    """
    rssm = model.rssm

    # ── Flatten start states: [B, T, ...] → [BT, ...] ─────────────────────
    B, T = start_states["deter"].shape[:2]
    BT   = B * T

    deter_0 = start_states["deter"].reshape(BT, -1).detach().to(device)
    stoch_0 = start_states["stoch"].reshape(BT, -1).detach().to(device)
    logits_0 = start_states["logits"].reshape(BT, rssm.stoch_dim,
                                               rssm.stoch_classes).detach().to(device)
    goal_0  = start_goals.reshape(BT, -1).detach().to(device)

    state = RSSMState(deter=deter_0, stoch=stoch_0, logits=logits_0)

    # ── Imagination rollout ────────────────────────────────────────────────
    feat_list   = []
    goal_list   = []
    action_list = []
    reward_list = []
    cont_list   = []

    for h in range(H):
        feat      = state.feature                             # [BT, state_dim]
        goal_h    = goal_0                                    # goal constant in imagination
        action, std = actor(feat, goal_h)                    # with gradient

        feat_list.append(feat)
        goal_list.append(goal_h)
        action_list.append(action)

        # Imagine next state (prior only — no observation)
        state = rssm.imagine_step(state, action)

        # Predict reward + continue from imagined state
        next_feat = state.feature
        rew  = model.reward_pred.predict(next_feat)          # [BT]
        cont = model.continue_pred.predict(next_feat)        # [BT] ∈ (0,1)
        reward_list.append(rew)
        cont_list.append(cont)

    # Stack: H × [BT, ...] → [H, BT, ...]
    feats   = torch.stack(feat_list,   dim=0)   # [H, BT, state_dim]
    goals   = torch.stack(goal_list,   dim=0)   # [H, BT, goal_dim]
    actions = torch.stack(action_list, dim=0)   # [H, BT, action_dim]
    rewards = torch.stack(reward_list, dim=0)   # [H, BT]
    conts   = torch.stack(cont_list,   dim=0)   # [H, BT]

    # Bootstrap value at final imagined state
    with torch.no_grad():
        v_final = critic.target(state.feature.detach(), goal_0)  # [BT]

    # ── λ-returns (Bellman targets) ────────────────────────────────────────
    # V_λ(s_H) = v_final
    # V_λ(s_t) = r_t + γ·c_t·((1-λ)·V(s_{t+1}) + λ·V_λ(s_{t+1}))
    with torch.no_grad():
        vals    = critic.target(feats.reshape(-1, feats.shape[-1]),
                                goals.reshape(-1, goals.shape[-1])
                                ).reshape(H, BT)     # [H, BT]

        targets = torch.zeros(H, BT, device=device)
        last    = v_final
        for h in reversed(range(H)):
            td = rewards[h] + gamma * conts[h] * last
            last = (1 - lam) * vals[h] + lam * td
            targets[h] = last

    targets = targets.detach()   # [H, BT]

    # ── Critic loss ────────────────────────────────────────────────────────
    pred_v = critic(feats.reshape(-1, feats.shape[-1]),
                    goals.reshape(-1, goals.shape[-1])
                    ).reshape(H, BT)                  # [H, BT]
    critic_loss = F.mse_loss(pred_v, targets)

    critic_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
    critic_opt.step()
    critic.update_target()

    # ── Actor loss ─────────────────────────────────────────────────────────
    # Recompute with gradient (separate forward pass to keep computation graph)
    feats_d = feats.detach()
    goals_d = goals.detach()

    act_means = []
    act_stds  = []
    for h in range(H):
        m, s = actor(feats_d[h], goals_d[h])
        act_means.append(m)
        act_stds.append(s)

    act_means = torch.stack(act_means, dim=0)   # [H, BT, action_dim]
    act_stds  = torch.stack(act_stds,  dim=0)

    # Policy gradient: maximise returns
    dist = torch.distributions.Normal(act_means, act_stds)
    log_probs = dist.log_prob(actions.detach()).sum(-1)    # [H, BT]
    norm_targets = (targets - targets.mean()) / (targets.std() + 1e-8)
    actor_loss_pg = -(log_probs * norm_targets).mean()

    # Entropy regularisation: keep policy from collapsing
    entropy = dist.entropy().sum(-1).mean()
    actor_loss = actor_loss_pg - entropy_scale * entropy

    actor_opt.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
    actor_opt.step()

    return {
        "actor":    float(actor_loss.item()),
        "critic":   float(critic_loss.item()),
        "entropy":  float(entropy.item()),
        "imagined_return": float(targets.mean().item()),
    }
