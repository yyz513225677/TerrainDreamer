"""Behavior cloning of the DreamerActor from human teleop demos.

Loads ``demos/demo_*.npz`` produced by ``scripts/teleop_record.py``, runs each
trajectory through the trained world model encoder + RSSM to recover posterior
latents, and trains the actor to match the recorded action distribution under
a Gaussian negative-log-likelihood loss. The world model itself is frozen by
default (the recorded demos are not yet in its training distribution and we'd
rather wait for joint RL fine-tuning to update it).

Usage::

    python3 scripts/bc_train.py \
        --in   checkpoints_auto/ckpt_latest.pt \
        --out  checkpoints_auto/ckpt_bc.pt \
        --demos demos/ \
        --epochs 30

After this writes ``ckpt_bc.pt`` you can resume RL fine-tuning with::

    ./run_auto.sh --resume checkpoints_auto/ckpt_bc.pt
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from terrain_dreamer.preprocessing.point_cloud_processor import PointCloudProcessor
from terrain_dreamer.envs.sensors.velodyne_vlp32 import PointCloud
from terrain_dreamer.world_model.terrain_dreamer_model import TerrainDreamerModel
from terrain_dreamer.world_model.dreamer_policy import DreamerActor


FEAT_DIM, ACTION_DIM, GOAL_DIM = 8, 2, 4
MAX_POINTS_BUFFER = 1024


def raw_to_feats(raw: np.ndarray, n: int, processor: PointCloudProcessor) -> np.ndarray:
    if n <= 0 or raw.size == 0:
        return np.zeros((MAX_POINTS_BUFFER, FEAT_DIM), dtype=np.float32)
    pc = PointCloud(timestamp=0.0, points=raw[:n].astype(np.float32))
    feats = processor.process(pc).features
    out = np.zeros((MAX_POINTS_BUFFER, FEAT_DIM), dtype=np.float32)
    m = min(feats.shape[0], MAX_POINTS_BUFFER)
    out[:m] = feats[:m]
    return out


def load_demos(demo_dir: Path, processor: PointCloudProcessor):
    """Load every demo, processing point clouds once. Returns list of dicts."""
    demos = []
    for path in sorted(demo_dir.glob("demo_*.npz")):
        d = np.load(path)
        T = d["action"].shape[0]
        if T < 2:
            continue
        feats = np.stack([
            raw_to_feats(d["points"][t], int(d["n_points"][t]), processor)
            for t in range(T)
        ]).astype(np.float32)
        demos.append({
            "feats":    feats,                            # [T, N, 8]
            "action":   d["action"].astype(np.float32),   # [T, 2]
            "goal_obs": d["goal_obs"].astype(np.float32), # [T, 4]
            "name":     path.name,
            "T":        T,
        })
        print(f"[bc] loaded {path.name}  T={T}")
    if not demos:
        raise RuntimeError(f"no demo_*.npz under {demo_dir}")
    return demos


def sample_batch(
    demos: List[dict],
    seq_len: int,
    batch_size: int,
    rng: np.random.Generator,
):
    """Sample fixed-length subsequences across demos."""
    feats_b, act_b, goal_b = [], [], []
    for _ in range(batch_size):
        d = demos[rng.integers(0, len(demos))]
        T = d["T"]
        if T <= seq_len:
            f = np.zeros((seq_len, MAX_POINTS_BUFFER, FEAT_DIM), dtype=np.float32)
            a = np.zeros((seq_len, ACTION_DIM), dtype=np.float32)
            g = np.zeros((seq_len, GOAL_DIM), dtype=np.float32)
            f[:T] = d["feats"]
            a[:T] = d["action"]
            g[:T] = d["goal_obs"]
        else:
            t0 = int(rng.integers(0, T - seq_len + 1))
            f = d["feats"][t0:t0 + seq_len]
            a = d["action"][t0:t0 + seq_len]
            g = d["goal_obs"][t0:t0 + seq_len]
        feats_b.append(f); act_b.append(a); goal_b.append(g)
    return (
        np.stack(feats_b),  # [B, T, N, 8]
        np.stack(act_b),    # [B, T, 2]
        np.stack(goal_b),   # [B, T, 4]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",     dest="src", required=True)
    ap.add_argument("--out",    dest="dst", required=True)
    ap.add_argument("--demos",  default="demos", type=Path)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--also-train-wm", action="store_true",
                    help="Also fine-tune the world model encoder/RSSM on demos "
                         "(default: frozen)")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[bc] device={device}")

    ck = torch.load(args.src, map_location=device)
    model = TerrainDreamerModel(
        input_channels=FEAT_DIM, embed_dim=256, action_dim=ACTION_DIM,
    ).to(device)
    model.load_state_dict(ck["model"])
    state_dim = model.rssm.deter_dim + model.rssm.stoch_total
    actor = DreamerActor(state_dim, goal_dim=GOAL_DIM, action_dim=ACTION_DIM).to(device)
    actor.load_state_dict(ck["actor"])

    if not args.also_train_wm:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)

    processor = PointCloudProcessor(voxel_size=0.15, max_points=MAX_POINTS_BUFFER)
    demos = load_demos(Path(args.demos), processor)
    rng = np.random.default_rng(0)

    for epoch in range(args.epochs):
        ep_nll, ep_n = 0.0, 0
        for _ in range(args.steps_per_epoch):
            feats_b, act_b, goal_b = sample_batch(
                demos, args.seq_len, args.batch_size, rng,
            )
            B, T = act_b.shape[:2]
            feats_t = torch.from_numpy(feats_b).to(device)
            act_t   = torch.from_numpy(act_b).to(device)
            goal_t  = torch.from_numpy(goal_b).to(device)

            BT = B * T
            N, C = feats_t.shape[2], feats_t.shape[3]
            flat = feats_t.reshape(BT, N, C)
            with torch.set_grad_enabled(args.also_train_wm):
                emb = model.encoder(flat).reshape(B, T, -1)
                _prior, post = model.rssm.observe_sequence(emb, act_t)
                feat_seq = torch.cat([post["deter"], post["stoch"]], dim=-1)

            mean, std = actor(
                feat_seq.reshape(BT, -1),
                goal_t.reshape(BT, -1),
            )
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(act_t.reshape(BT, -1)).sum(-1)
            nll = -log_probs.mean()

            actor_opt.zero_grad()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            actor_opt.step()

            ep_nll += float(nll.item())
            ep_n += 1

        print(f"[bc] epoch {epoch+1:3d}/{args.epochs}  "
              f"nll={ep_nll/max(ep_n,1):.4f}")

    ck["actor"] = actor.state_dict()
    if args.also_train_wm:
        ck["model"] = model.state_dict()
    torch.save(ck, args.dst)
    print(f"[bc] wrote {args.dst}")


if __name__ == "__main__":
    main()
