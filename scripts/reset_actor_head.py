"""Reset DreamerActor (and optionally DreamerCritic) weights in a checkpoint.

The world model is preserved. Used to break a collapsed policy basin while
keeping the trained representation. The combined actor+critic reset is needed
when the critic has internalized a degenerate value landscape (e.g. all
trajectories end in flip → critic predicts uniform negative return) — in that
case resetting the actor alone leaves the contaminated value baseline that
steers the fresh actor right back into the same basin.

Usage::

    # actor only (legacy)
    python scripts/reset_actor_head.py \
        --in  checkpoints_auto/ckpt_latest.pt \
        --out checkpoints_auto/ckpt_actor_reset.pt

    # actor + critic
    python scripts/reset_actor_head.py --reset-critic \
        --in  checkpoints_auto/ckpt_latest.pt \
        --out checkpoints_auto/ckpt_ac_reset.pt

    # critic only (keep actor — useful when a previously-good actor sits on
    # a poisoned critic, e.g. after a curriculum-induced collapse)
    python scripts/reset_actor_head.py --keep-actor --reset-critic \
        --in  checkpoints_auto/ckpt_m00075.pt \
        --out checkpoints_auto/ckpt_critic_reset.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from terrain_dreamer.world_model.terrain_dreamer_model import TerrainDreamerModel
from terrain_dreamer.world_model.dreamer_policy import DreamerActor, DreamerCritic


FEAT_DIM, ACTION_DIM, GOAL_DIM = 8, 2, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--reset-critic", action="store_true",
                    help="Also re-initialize the critic (recommended after collapse)")
    ap.add_argument("--keep-actor", action="store_true",
                    help="Don't replace the actor (only meaningful with "
                         "--reset-critic: useful when the actor at peak is "
                         "good but the critic has been poisoned afterwards)")
    args = ap.parse_args()
    if args.keep_actor and not args.reset_critic:
        ap.error("--keep-actor is only useful with --reset-critic; otherwise "
                 "this script would copy the input unchanged.")

    ck = torch.load(args.src, map_location="cpu")
    print(f"[reset] loaded {args.src}  keys={list(ck.keys())}")

    model = TerrainDreamerModel(
        input_channels=FEAT_DIM, embed_dim=256, action_dim=ACTION_DIM,
    )
    model.load_state_dict(ck["model"])
    state_dim = model.rssm.deter_dim + model.rssm.stoch_total

    if args.keep_actor:
        print(f"[reset] keeping actor (state_dim={state_dim})")
    else:
        fresh_actor = DreamerActor(state_dim, goal_dim=GOAL_DIM,
                                     action_dim=ACTION_DIM)
        ck["actor"] = fresh_actor.state_dict()
        print(f"[reset] replaced actor (state_dim={state_dim})")

    if args.reset_critic:
        fresh_critic = DreamerCritic(state_dim, goal_dim=GOAL_DIM)
        ck["critic"] = fresh_critic.state_dict()
        print(f"[reset] replaced critic (state_dim={state_dim})")

    torch.save(ck, args.dst)
    print(f"[reset] wrote {args.dst}")


if __name__ == "__main__":
    main()
