#!/usr/bin/env python3
"""
train_dreamer_auto.py — Autonomous Dreamer training loop on Gazebo + Jackal.

Rotates the (x, y) goal after each completed mission, records the outbound
path, retraces it on return, and feeds the combined trajectory to the
DreamerV3-style world model.

Prerequisites (handled by run_auto.sh):
    1. /opt/ros/noetic/setup.bash sourced
    2. ros_ws/devel/setup.bash sourced
    3. JACKAL_URDF_EXTRAS exported
    4. `roslaunch terrain_dreamer_bringup moon_jackal.launch` running

This script ONLY drives the rover and trains the model — it assumes Gazebo is
already up.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from terrain_dreamer.envs.ros_jackal_env import RosJackalEnv
from terrain_dreamer.preprocessing.point_cloud_processor import (
    PointCloudProcessor,
)
from terrain_dreamer.envs.sensors.velodyne_vlp32 import PointCloud
from terrain_dreamer.training.dreamer_buffer import DreamerReplayBuffer
from terrain_dreamer.world_model.terrain_dreamer_model import TerrainDreamerModel
from terrain_dreamer.world_model.dreamer_policy import (
    DreamerActor,
    DreamerCritic,
    imagine_train,
)
from terrain_dreamer.world_model.rssm import RSSMState


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_POINTS_BUFFER = 1024   # N in features [T, N, 8] — matches buffer default
FEAT_DIM = 8
ACTION_DIM = 2
GOAL_DIM = 4

# Curriculum
INIT_GOAL_DIST = 4.0       # m — start easy
MAX_GOAL_DIST  = 25.0      # m — eventual reach
CURR_WIN       = 20        # episodes in the rolling-success window
CURR_UP_TH     = 0.7       # grow radius if success-rate > this
CURR_DOWN_TH   = 0.3       # shrink radius if success-rate < this
CURR_STEP_M    = 1.0

# Return phase
RETURN_LOOKAHEAD = 1.5     # m, pure-pursuit lookahead
RETURN_MIN_SPACING = 0.3   # m, resample the outbound path to this spacing
RETURN_ARRIVE_DIST = 0.8


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resample_path(path: np.ndarray, spacing: float) -> np.ndarray:
    """Downsample an (N, 2) path to approximately `spacing` meters between points."""
    if len(path) < 2:
        return path
    out = [path[0]]
    for p in path[1:]:
        if np.linalg.norm(p - out[-1]) >= spacing:
            out.append(p)
    if not np.allclose(out[-1], path[-1]):
        out.append(path[-1])
    return np.asarray(out, dtype=np.float32)


def pure_pursuit_action(
    pose: np.ndarray,          # (x, y, yaw)
    path: np.ndarray,          # (K, 2)
    path_idx: int,
    lookahead: float,
) -> Tuple[np.ndarray, int]:
    """Classic pure-pursuit: drive toward point ~lookahead m ahead on the path.
    Returns (action ∈ [-1, 1]², new path_idx)."""
    K = len(path)
    while path_idx < K - 1:
        if np.linalg.norm(path[path_idx] - pose[:2]) >= lookahead:
            break
        path_idx += 1

    target = path[path_idx]
    dx = target[0] - pose[0]
    dy = target[1] - pose[1]
    bearing = math.atan2(dy, dx)
    heading_err = math.atan2(math.sin(bearing - pose[2]),
                              math.cos(bearing - pose[2]))
    # stop-and-turn if badly misaligned, otherwise forward + proportional steer
    if abs(heading_err) > math.radians(45):
        ang = np.sign(heading_err) * 1.0
        lin = 0.0
    else:
        ang = np.clip(heading_err * 1.5, -1.0, 1.0)
        lin = float(np.cos(heading_err)) * 0.8
    return np.array([lin, ang], dtype=np.float32), path_idx


def stop_and_turn_exploration(goal_obs: np.ndarray, noise_std: float) -> np.ndarray:
    """Heuristic exploratory action used before the actor has learned.
    Uses heading_err_norm = goal_obs[3] (∈ [-1, 1] = angle / π)."""
    heading_err = goal_obs[3] * math.pi
    if abs(heading_err) > math.radians(35):
        lin = 0.0
        ang = float(np.sign(heading_err)) * 0.9
    else:
        lin = 0.7 + np.random.randn() * noise_std
        ang = np.clip(heading_err * 1.2 + np.random.randn() * noise_std,
                      -1.0, 1.0)
    return np.clip(np.array([lin, ang], dtype=np.float32), -1.0, 1.0)


# ---------------------------------------------------------------------------
# Features: raw PointCloud2 points → processor features [N_pad, 8]
# ---------------------------------------------------------------------------
def raw_points_to_features(
    raw_xyzi: np.ndarray,       # [N, 4]
    processor: PointCloudProcessor,
    n_valid: int,
    pad_to: int,
) -> np.ndarray:
    """Run the PointCloudProcessor on a raw scan and pad/truncate to pad_to rows."""
    if n_valid <= 0 or raw_xyzi.size == 0:
        return np.zeros((pad_to, FEAT_DIM), dtype=np.float32)

    pc = PointCloud(timestamp=0.0, points=raw_xyzi[:n_valid].astype(np.float32))
    proc = processor.process(pc)
    feats = proc.features  # [M, 8]
    out = np.zeros((pad_to, FEAT_DIM), dtype=np.float32)
    m = min(feats.shape[0], pad_to)
    out[:m] = feats[:m]
    return out


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------
@dataclass
class EpisodeBuffer:
    features: List[np.ndarray] = field(default_factory=list)
    actions:  List[np.ndarray] = field(default_factory=list)
    rewards:  List[float]      = field(default_factory=list)
    continues:List[float]      = field(default_factory=list)
    goal_obs: List[np.ndarray] = field(default_factory=list)
    path_xy:  List[np.ndarray] = field(default_factory=list)  # for return phase

    def __len__(self) -> int:
        return len(self.actions)

    def finalize(self) -> Optional[Dict[str, np.ndarray]]:
        if len(self.actions) == 0:
            return None
        return {
            "features":  np.stack(self.features,  axis=0),
            "actions":   np.stack(self.actions,   axis=0),
            "rewards":   np.asarray(self.rewards,   dtype=np.float32),
            "continues": np.asarray(self.continues, dtype=np.float32),
            "goal_obs":  np.stack(self.goal_obs,  axis=0),
            "path_xy":   np.stack(self.path_xy,   axis=0),
        }


def run_phase(
    env: RosJackalEnv,
    processor: PointCloudProcessor,
    actor: Optional[DreamerActor],
    device: torch.device,
    *,
    spawn: Tuple[float, float, float],
    goal: Tuple[float, float],
    max_steps: int,
    use_exploration: bool,
    explore_noise: float,
    use_actor_prob: float,
    state: Optional[RSSMState] = None,
    model: Optional[TerrainDreamerModel] = None,
) -> Tuple[EpisodeBuffer, Dict, RSSMState]:
    """Run a single GOING phase. Returns (episode buffer, info, final RSSM state).

    The posterior RSSM state is carried forward step-by-step so the actor sees a
    real recurrent context. If `model` is None, the actor is bypassed and we
    fall back to the stop-and-turn heuristic.
    """
    ep = EpisodeBuffer()
    obs, reset_info = env.reset(options={
        "spawn_x": spawn[0], "spawn_y": spawn[1], "spawn_yaw": spawn[2],
        "goal": goal,
    })
    # The env may have nudged the spawn to find a level spot. Use that actual
    # (x, y) as the "home" target for the return phase, not the requested one.
    actual_spawn_xy = reset_info.get("spawn_xy", np.array(spawn[:2], dtype=np.float32))

    # Initialize RSSM state if we have a model
    if model is not None:
        rssm = model.rssm
        state = RSSMState(
            deter=torch.zeros(1, rssm.deter_dim,  device=device),
            stoch=torch.zeros(1, rssm.stoch_total, device=device),
            logits=torch.zeros(1, rssm.stoch_dim, rssm.stoch_classes, device=device),
        )
        prev_action = torch.zeros(1, ACTION_DIM, device=device)

    total_reward = 0.0
    reached = False
    flipped = False
    for t in range(max_steps):
        # Encode current obs + update posterior state
        feats_np = raw_points_to_features(
            obs["points"], processor, int(obs["n_points"]),
            pad_to=MAX_POINTS_BUFFER,
        )

        # Decide action ------------------------------------------------------
        use_heuristic = (
            actor is None or model is None
            or (use_exploration and random.random() > use_actor_prob)
        )
        if use_heuristic:
            action = stop_and_turn_exploration(obs["goal_obs"], explore_noise)
        else:
            with torch.no_grad():
                pts_t = torch.from_numpy(feats_np[None]).to(device)   # [1, N, 8]
                embed = model.encoder(pts_t)                          # [1, embed_dim]
                _prior, post = model.rssm.observe_step(
                    state, prev_action, embed,
                )
                state = post
                goal_t = torch.from_numpy(obs["goal_obs"][None]).to(device)
                a_t = actor.act(state.feature, goal_t,
                                explore=use_exploration,
                                explore_scale=explore_noise)
                action = a_t[0].cpu().numpy().astype(np.float32)

        # Step the env -------------------------------------------------------
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        ep.features .append(feats_np)
        ep.actions  .append(action.copy())
        ep.rewards  .append(float(reward))
        ep.continues.append(0.0 if terminated else 1.0)
        ep.goal_obs .append(obs["goal_obs"].copy())
        ep.path_xy  .append(obs["pose"][:2].copy())

        if model is not None and not use_heuristic:
            prev_action = torch.from_numpy(action[None]).to(device)

        obs = next_obs
        if info.get("reached"):
            reached = True
        if info.get("flipped"):
            flipped = True
        if done:
            break

    info_out = {
        "steps": len(ep),
        "reward": total_reward,
        "reached": reached,
        "flipped": flipped,
        "final_pose": obs["pose"].copy(),
        "final_goal_dist": info.get("dist_to_goal", float("nan")),
        "actual_spawn_xy": actual_spawn_xy,
    }
    return ep, info_out, state


# ---------------------------------------------------------------------------
# HER: relabel a failed GOING episode with its own reached endpoint
# ---------------------------------------------------------------------------
def her_relabel(ep: EpisodeBuffer, new_goal: np.ndarray) -> EpisodeBuffer:
    """Recompute goal_obs / rewards / continues as if `new_goal` had been the goal."""
    out = EpisodeBuffer()
    prev_dist = None
    poses = ep.path_xy
    for t, pose in enumerate(poses):
        dx = new_goal[0] - pose[0]
        dy = new_goal[1] - pose[1]
        dist = math.hypot(dx, dy)
        # We don't have yaw in path_xy — approximate heading_err as 0.
        goal_obs = np.array([
            np.clip(dx / 30.0, -1.0, 1.0),
            np.clip(dy / 30.0, -1.0, 1.0),
            np.clip(dist / 30.0, 0.0, 1.0),
            0.0,
        ], dtype=np.float32)

        shaping = 0.0 if prev_dist is None else (prev_dist - dist) * 2.0
        prev_dist = dist
        reached = dist < 0.8 and t == len(poses) - 1
        reward = shaping + (25.0 if reached else 0.0)
        cont = 0.0 if reached else 1.0

        out.features .append(ep.features[t])
        out.actions  .append(ep.actions[t])
        out.rewards  .append(float(reward))
        out.continues.append(float(cont))
        out.goal_obs .append(goal_obs)
        out.path_xy  .append(pose.copy())
    return out


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------
def train_step(
    model: TerrainDreamerModel,
    actor: DreamerActor,
    critic: DreamerCritic,
    buffer: DreamerReplayBuffer,
    model_opt: torch.optim.Optimizer,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    imagine_horizon: int,
) -> Dict[str, float]:
    batch = buffer.sample(batch_size)
    feats_np = batch["features"]   # [B, T, N, 8]
    acts_np  = batch["actions"]
    rews_np  = batch["rewards"]
    conts_np = batch["continues"]
    goal_np  = batch["goal_obs"]

    feats_t = torch.from_numpy(feats_np).to(device)
    acts_t  = torch.from_numpy(acts_np).to(device)
    rews_t  = torch.from_numpy(rews_np).to(device)
    conts_t = torch.from_numpy(conts_np).to(device)
    goal_t  = torch.from_numpy(goal_np).to(device)

    # --- World model loss ---
    losses = model.training_loss(feats_t, acts_t, rews_t, conts_t)
    model_opt.zero_grad()
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
    model_opt.step()

    # --- Posteriors for imagination start states ---
    with torch.no_grad():
        B, T = acts_t.shape[:2]
        BT = B * T
        N, C = feats_t.shape[2], feats_t.shape[3]
        embeds = model.encoder(feats_t.reshape(BT, N, C)).reshape(B, T, -1)
        _, post = model.rssm.observe_sequence(embeds, acts_t)

    ac_losses = imagine_train(
        model, actor, critic,
        actor_opt, critic_opt,
        start_states=post,
        start_goals=goal_t,
        device=device,
        H=imagine_horizon,
    )

    out = {f"wm/{k}": float(v.item() if torch.is_tensor(v) else v)
           for k, v in losses.items()}
    out.update({f"ac/{k}": v for k, v in ac_losses.items()})
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--total_missions", type=int, default=500)
    ap.add_argument("--max_steps",      type=int, default=600,
                    help="max steps per GOING or RETURNING phase")
    ap.add_argument("--train_every_episodes", type=int, default=1)
    ap.add_argument("--updates_per_train",    type=int, default=4)
    ap.add_argument("--batch_size",  type=int, default=16)
    ap.add_argument("--seq_len",     type=int, default=16)
    ap.add_argument("--imagine_h",   type=int, default=15)
    ap.add_argument("--warmup_missions",    type=int, default=10,
                    help="collect this many episodes with the heuristic before "
                         "engaging the actor")
    ap.add_argument("--use_actor_prob_start", type=float, default=0.0)
    ap.add_argument("--use_actor_prob_end",   type=float, default=0.85)
    ap.add_argument("--checkpoint_dir", default="checkpoints_auto")
    ap.add_argument("--resume", default=None, help="path to .pt to resume from")
    ap.add_argument("--log_path", default=None,
                    help="CSV log path (default: <checkpoint_dir>/training_log.csv)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path) if args.log_path else ckpt_dir / "training_log.csv"

    # ---------------------------------------------------------------
    # Env + preprocessor
    # ---------------------------------------------------------------
    print("[init] Connecting to ROS / Gazebo …")
    env = RosJackalEnv(max_episode_steps=args.max_steps)
    env.wait_ready(timeout=60.0)
    print("[init] Env ready.")

    processor = PointCloudProcessor(
        voxel_size=0.15,
        max_points=MAX_POINTS_BUFFER,
    )

    # ---------------------------------------------------------------
    # Model + actor-critic
    # ---------------------------------------------------------------
    model = TerrainDreamerModel(
        input_channels=FEAT_DIM,
        embed_dim=256,
        action_dim=ACTION_DIM,
    ).to(device)
    state_dim = model.rssm.deter_dim + model.rssm.stoch_total
    actor  = DreamerActor (state_dim, goal_dim=GOAL_DIM, action_dim=ACTION_DIM).to(device)
    critic = DreamerCritic(state_dim, goal_dim=GOAL_DIM).to(device)

    model_opt  = torch.optim.Adam(model.parameters(),  lr=1e-4)
    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=8e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=8e-5)

    buffer = DreamerReplayBuffer(
        seq_len=args.seq_len,
        max_points=MAX_POINTS_BUFFER,
        feat_dim=FEAT_DIM,
        action_dim=ACTION_DIM,
        max_episodes=500,
        min_episodes=3,
    )

    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model .load_state_dict(ck["model"])
        actor .load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        print(f"[init] Resumed from {args.resume}")

    # ---------------------------------------------------------------
    # CSV log
    # ---------------------------------------------------------------
    log_new = not log_path.exists()
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if log_new:
        log.writerow([
            "mission", "phase", "steps", "reward", "reached", "flipped",
            "goal_dist_max", "success_rate", "wm_total", "actor_loss",
            "critic_loss", "imagined_return",
        ])

    # ---------------------------------------------------------------
    # Curriculum state
    # ---------------------------------------------------------------
    goal_dist_max = INIT_GOAL_DIST
    success_hist: List[int] = []

    rng = np.random.default_rng(args.seed)

    # ---------------------------------------------------------------
    # Graceful shutdown
    # ---------------------------------------------------------------
    stop_flag = {"v": False}
    def _handle_sigint(sig, frame):
        print("\n[signal] SIGINT — saving and exiting…")
        stop_flag["v"] = True
    signal.signal(signal.SIGINT, _handle_sigint)

    # ---------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------
    last_losses: Dict[str, float] = {}
    mission = 0
    try:
        while mission < args.total_missions and not stop_flag["v"]:
            mission += 1

            # -------- Pick a new goal (rotation) ---------------------------
            # Uniform random point on an annulus [0.6*dmax, dmax], random bearing.
            radius = float(rng.uniform(0.6 * goal_dist_max, goal_dist_max))
            theta  = float(rng.uniform(-math.pi, math.pi))
            goal = (radius * math.cos(theta), radius * math.sin(theta))
            spawn = (0.0, 0.0, float(rng.uniform(-math.pi, math.pi)))

            # Anneal actor usage
            frac = min(mission / max(args.warmup_missions * 4, 1), 1.0)
            use_actor_prob = (
                args.use_actor_prob_start
                + frac * (args.use_actor_prob_end - args.use_actor_prob_start)
            )
            use_model = mission > args.warmup_missions

            # -------- GOING phase ------------------------------------------
            going, going_info, _ = run_phase(
                env, processor,
                actor=actor if use_model else None,
                model=model if use_model else None,
                device=device,
                spawn=spawn, goal=goal,
                max_steps=args.max_steps,
                use_exploration=True,
                explore_noise=max(0.1, 0.6 * (1.0 - frac)),
                use_actor_prob=use_actor_prob,
            )

            success_hist.append(1 if going_info["reached"] else 0)
            success_hist = success_hist[-CURR_WIN:]
            success_rate = float(np.mean(success_hist)) if success_hist else 0.0

            log.writerow([
                mission, "going", going_info["steps"], f"{going_info['reward']:.3f}",
                int(going_info["reached"]), int(going_info["flipped"]),
                f"{goal_dist_max:.2f}", f"{success_rate:.3f}",
                last_losses.get("wm/total", ""),
                last_losses.get("ac/actor", ""),
                last_losses.get("ac/critic", ""),
                last_losses.get("ac/imagined_return", ""),
            ])
            log_f.flush()

            print(f"[m{mission:04d} going]  goal=({goal[0]:+5.1f},{goal[1]:+5.1f})  "
                  f"steps={going_info['steps']:3d}  reward={going_info['reward']:+7.2f}  "
                  f"reached={int(going_info['reached'])}  flipped={int(going_info['flipped'])}  "
                  f"dmax={goal_dist_max:.1f}  succ={success_rate:.2f}")

            # -------- Push GOING to buffer ---------------------------------
            ep_data = going.finalize()
            if ep_data is not None:
                buffer.add_episode(
                    features  = ep_data["features"],
                    actions   = ep_data["actions"],
                    rewards   = ep_data["rewards"],
                    continues = ep_data["continues"],
                    goal_obs  = ep_data["goal_obs"],
                )

            # -------- HER relabel on failure -------------------------------
            if (not going_info["reached"]) and len(going) > 4:
                final_xy = going.path_xy[-1]
                if np.linalg.norm(final_xy) > 1.0:
                    her_ep = her_relabel(going, final_xy.astype(np.float32))
                    her_data = her_ep.finalize()
                    if her_data is not None:
                        buffer.add_episode(
                            features  = her_data["features"],
                            actions   = her_data["actions"],
                            rewards   = her_data["rewards"],
                            continues = her_data["continues"],
                            goal_obs  = her_data["goal_obs"],
                        )

            # -------- RETURNING phase --------------------------------------
            if going_info["reached"]:
                home_xy = going_info["actual_spawn_xy"]
                path = np.stack(going.path_xy, axis=0)
                retrace = resample_path(path[::-1], RETURN_MIN_SPACING)

                # Last obs (position after reaching goal) → follow retrace.
                obs = env._make_obs()
                path_idx = 0
                ret = EpisodeBuffer()
                ret_reached = False
                ret_flipped = False
                for t in range(args.max_steps):
                    pose = obs["pose"]
                    dist_home = float(np.linalg.norm(pose[:2] - home_xy))
                    if dist_home < RETURN_ARRIVE_DIST:
                        ret_reached = True
                        break

                    action, path_idx = pure_pursuit_action(
                        pose, retrace, path_idx, RETURN_LOOKAHEAD,
                    )
                    # Track features for the buffer
                    feats_np = raw_points_to_features(
                        obs["points"], processor, int(obs["n_points"]),
                        pad_to=MAX_POINTS_BUFFER,
                    )
                    next_obs, reward, terminated, truncated, info = env.step(action)

                    # Rewrite goal_obs so the buffer sample is consistent
                    dx = home_xy[0] - pose[0]; dy = home_xy[1] - pose[1]
                    home_dist = math.hypot(dx, dy)
                    g_obs = np.array([
                        np.clip(dx / 30.0, -1, 1),
                        np.clip(dy / 30.0, -1, 1),
                        np.clip(home_dist / 30.0, 0, 1),
                        0.0,
                    ], dtype=np.float32)

                    ret.features .append(feats_np)
                    ret.actions  .append(action)
                    ret.rewards  .append(float(reward))
                    ret.continues.append(0.0 if terminated else 1.0)
                    ret.goal_obs .append(g_obs)
                    ret.path_xy  .append(pose[:2].copy())

                    obs = next_obs
                    if info.get("flipped"):
                        ret_flipped = True; break
                    if terminated or truncated:
                        break

                log.writerow([
                    mission, "return", len(ret),
                    f"{sum(ret.rewards):.3f}" if len(ret) else "0",
                    int(ret_reached), int(ret_flipped),
                    f"{goal_dist_max:.2f}", f"{success_rate:.3f}",
                    "", "", "", "",
                ])
                log_f.flush()

                print(f"[m{mission:04d} return] steps={len(ret):3d} "
                      f"home={int(ret_reached)} flipped={int(ret_flipped)}")

                ret_data = ret.finalize()
                if ret_data is not None:
                    buffer.add_episode(
                        features  = ret_data["features"],
                        actions   = ret_data["actions"],
                        rewards   = ret_data["rewards"],
                        continues = ret_data["continues"],
                        goal_obs  = ret_data["goal_obs"],
                    )

            # -------- Curriculum update ------------------------------------
            if len(success_hist) >= CURR_WIN:
                if success_rate > CURR_UP_TH and goal_dist_max < MAX_GOAL_DIST:
                    goal_dist_max = min(MAX_GOAL_DIST, goal_dist_max + CURR_STEP_M)
                    print(f"[curr] +{CURR_STEP_M} → dmax={goal_dist_max:.1f}")
                elif success_rate < CURR_DOWN_TH and goal_dist_max > INIT_GOAL_DIST:
                    goal_dist_max = max(INIT_GOAL_DIST, goal_dist_max - CURR_STEP_M)
                    print(f"[curr] -{CURR_STEP_M} → dmax={goal_dist_max:.1f}")

            # -------- Training ---------------------------------------------
            if buffer.ready() and mission % args.train_every_episodes == 0:
                for _ in range(args.updates_per_train):
                    last_losses = train_step(
                        model, actor, critic, buffer,
                        model_opt, actor_opt, critic_opt,
                        device, args.batch_size, args.imagine_h,
                    )
                print(f"[train] wm={last_losses.get('wm/total', 0):.3f} "
                      f"kl={last_losses.get('wm/kl', 0):.3f} "
                      f"actor={last_losses.get('ac/actor', 0):.3f} "
                      f"critic={last_losses.get('ac/critic', 0):.3f} "
                      f"ret_hat={last_losses.get('ac/imagined_return', 0):.2f}")

            # -------- Checkpoint -------------------------------------------
            if mission % 25 == 0:
                ck_path = ckpt_dir / f"ckpt_m{mission:05d}.pt"
                torch.save({
                    "model":  model.state_dict(),
                    "actor":  actor.state_dict(),
                    "critic": critic.state_dict(),
                    "mission": mission,
                    "goal_dist_max": goal_dist_max,
                }, ck_path)
                print(f"[save] {ck_path}")

    finally:
        log_f.close()
        ck_path = ckpt_dir / "ckpt_latest.pt"
        torch.save({
            "model":  model.state_dict(),
            "actor":  actor.state_dict(),
            "critic": critic.state_dict(),
            "mission": mission,
            "goal_dist_max": goal_dist_max,
        }, ck_path)
        print(f"[save] {ck_path}")
        env.close()


if __name__ == "__main__":
    main()
