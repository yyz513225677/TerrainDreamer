#!/usr/bin/env python3
"""
generate_heightmaps.py — Procedural Moon terrain heightmaps for Gazebo.

Produces one 513x513 grayscale PNG per environment (Gazebo heightmap needs
(2^N)+1 square). Each env has its own fBm profile + feature set.

Output files land in worlds/heightmaps/<env>.png alongside this script's
package. Each world file references its PNG via model://.

Usage:
    rosrun terrain_dreamer_bringup generate_heightmaps.py
    # or
    python3 ros_ws/src/terrain_dreamer_bringup/scripts/generate_heightmaps.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


RES = 513  # (2^9)+1 required by Gazebo heightmap
SIZE_M = 100.0  # world-side length in meters (± 50 m)


def fbm(shape, octaves: int, hurst: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H, W = shape
    acc = np.zeros(shape, dtype=np.float32)
    amp = 1.0
    freq = 1.0
    for _ in range(octaves):
        fy = int(max(2, H * freq / 4))
        fx = int(max(2, W * freq / 4))
        noise = rng.standard_normal((fy, fx)).astype(np.float32)
        upscaled = np.array(
            Image.fromarray(noise).resize((W, H), Image.BICUBIC),
            dtype=np.float32,
        )
        acc += amp * upscaled
        amp *= 2 ** -(hurst + 1) / 2
        freq *= 2.0
    acc -= acc.mean()
    std = acc.std()
    return acc / (std + 1e-8)


def add_craters(h: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 100)
    H, W = h.shape
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(count):
        cx = rng.integers(20, W - 20)
        cy = rng.integers(20, H - 20)
        r = rng.integers(8, 45)
        depth = 0.22 * r
        rim = 0.06 * r
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        bowl = -depth * np.clip(1 - (d / r) ** 2, 0, 1)
        rim_band = rim * np.exp(-((d - r) ** 2) / (0.08 * r * r + 1e-3))
        h += bowl + rim_band
    return h


def add_mesas(h: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 200)
    H, W = h.shape
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(count):
        cx = rng.integers(40, W - 40)
        cy = rng.integers(40, H - 40)
        r = rng.integers(25, 70)
        height = rng.uniform(2.0, 6.0)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        plateau = height / (1 + np.exp((d - r) * 0.4))
        h += plateau
    return h


def add_rille(h: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 300)
    H, W = h.shape
    t = np.linspace(0, 1, 400)
    cx = W * (0.15 + 0.7 * t) + 12 * np.sin(6 * t)
    cy = H * (0.25 + 0.5 * t) + 8 * np.cos(4 * t)
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(len(t)):
        d = np.sqrt((xx - cx[i]) ** 2 + (yy - cy[i]) ** 2)
        h -= 2.5 * np.exp(-(d ** 2) / 12.0)
    return h


ENV_CONFIGS = {
    "mare":          dict(oct=6, hurst=0.76, crater=4,  mesa=0,  rille=False, amp=0.6),
    "highland":      dict(oct=7, hurst=0.95, crater=8,  mesa=3,  rille=False, amp=2.5),
    "cratered":      dict(oct=6, hurst=0.82, crater=25, mesa=0,  rille=False, amp=1.2),
    "boulder_field": dict(oct=6, hurst=0.80, crater=6,  mesa=0,  rille=False, amp=0.9),
    "rille":         dict(oct=6, hurst=0.78, crater=5,  mesa=0,  rille=True,  amp=1.0),
}


def to_png(h: np.ndarray) -> Image.Image:
    lo, hi = h.min(), h.max()
    norm = (h - lo) / (hi - lo + 1e-9)
    img = (norm * 255).astype(np.uint8)
    return Image.fromarray(img, mode="L")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output dir (default: package worlds/heightmaps/)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.out is None:
        here = Path(__file__).resolve().parent.parent
        args.out = here / "worlds" / "heightmaps"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for env, cfg in ENV_CONFIGS.items():
        base = cfg["amp"] * fbm((RES, RES), cfg["oct"], cfg["hurst"], args.seed)
        if cfg["crater"]:
            base = add_craters(base, cfg["crater"], args.seed)
        if cfg["mesa"]:
            base = add_mesas(base, cfg["mesa"], args.seed)
        if cfg["rille"]:
            base = add_rille(base, args.seed)
        png = to_png(base)
        path = out / f"{env}.png"
        png.save(path)
        stats_lo, stats_hi = base.min(), base.max()
        print(f"  [{env:>14s}] {path.name}  range=[{stats_lo:+.2f}, {stats_hi:+.2f}] m")

    meta = out / "metadata.yaml"
    meta.write_text(
        f"# Heightmap metadata for Gazebo <heightmap> blocks\n"
        f"resolution: {RES}\n"
        f"world_size_m: {SIZE_M}\n"
    )
    print(f"\n  Wrote {len(ENV_CONFIGS)} heightmaps to {out}")


if __name__ == "__main__":
    main()
