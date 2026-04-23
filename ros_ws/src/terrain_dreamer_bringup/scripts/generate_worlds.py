#!/usr/bin/env python3
"""
generate_worlds.py — Emit one Gazebo .world file per lunar environment.

Each world:
  - Moon gravity -1.62 m/s^2
  - Sun + ambient tuned for lunar lighting (strong directional, low ambient)
  - Heightmap terrain referencing worlds/heightmaps/<env>.png
  - Env-specific boulder scatter (placed as simple <model> spheres of rock)

Run AFTER generate_heightmaps.py. Together they rebuild the whole world set.

Usage:
    python3 ros_ws/src/terrain_dreamer_bringup/scripts/generate_worlds.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


WORLD_SIZE_M = 100.0
HEIGHT_Z_SCALE_M = 10.0  # PNG [0,255] → [0, 10 m]


ENV_ROCKS = {
    "mare":          dict(n=6,   size_mu=0.30, size_sigma=0.10),
    "highland":      dict(n=40,  size_mu=0.55, size_sigma=0.25),
    "cratered":      dict(n=25,  size_mu=0.45, size_sigma=0.20),
    "boulder_field": dict(n=120, size_mu=0.60, size_sigma=0.30),
    "rille":         dict(n=15,  size_mu=0.40, size_sigma=0.15),
}


HEADER = """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="moon_{env}">

    <physics type="ode">
      <!-- 250 Hz * 0.004 step = 1.0 real-time target. VLP-32 @ 10 Hz needs
           CPU headroom; going faster makes the sensor plugin miss scans. -->
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
      <gravity>0 0 -1.62</gravity>
    </physics>

    <scene>
      <ambient>0.08 0.08 0.10 1.0</ambient>
      <background>0.01 0.01 0.02 1.0</background>
      <shadows>true</shadows>
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 80 0 0.9 0</pose>
      <diffuse>1.0 0.95 0.85 1</diffuse>
      <specular>0.3 0.3 0.3 1</specular>
      <direction>0.3 0.2 -0.9</direction>
    </light>

    <model name="terrain">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <heightmap>
              <uri>file://{heightmap_path}</uri>
              <size>{sx:.2f} {sy:.2f} {sz:.2f}</size>
              <pos>0 0 0</pos>
            </heightmap>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <heightmap>
              <uri>file://{heightmap_path}</uri>
              <size>{sx:.2f} {sy:.2f} {sz:.2f}</size>
              <pos>0 0 0</pos>
              <texture>
                <diffuse>file://media/materials/textures/dirt_diffusespecular.png</diffuse>
                <normal>file://media/materials/textures/flat_normal.png</normal>
                <size>20</size>
              </texture>
            </heightmap>
          </geometry>
        </visual>
      </link>
    </model>
"""


ROCK_TEMPLATE = """
    <model name="rock_{i:03d}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.3f}</pose>
      <link name="link">
        <collision name="c">
          <geometry><sphere><radius>{r:.3f}</radius></sphere></geometry>
        </collision>
        <visual name="v">
          <geometry><sphere><radius>{r:.3f}</radius></sphere></geometry>
          <material>
            <ambient>0.22 0.20 0.18 1</ambient>
            <diffuse>0.35 0.32 0.28 1</diffuse>
            <specular>0.05 0.05 0.05 1</specular>
          </material>
        </visual>
      </link>
    </model>
"""


FOOTER = """
    <gui>
      <camera name="user_camera">
        <pose>-15 -15 12 0 0.55 0.78</pose>
      </camera>
    </gui>

  </world>
</sdf>
"""


def generate_rocks(env: str, seed: int, n: int, size_mu: float, size_sigma: float) -> str:
    mu, sigma = size_mu, size_sigma
    rng = np.random.default_rng(seed + hash(env) % 10000)
    half = WORLD_SIZE_M / 2 - 2
    parts = []
    for i in range(n):
        x = rng.uniform(-half, half)
        y = rng.uniform(-half, half)
        r = float(np.clip(rng.normal(mu, sigma), 0.1, 1.2))
        z = r * 0.3  # partly embedded
        yaw = rng.uniform(0, 2 * np.pi)
        parts.append(ROCK_TEMPLATE.format(i=i, x=x, y=y, z=z, r=r, yaw=yaw))
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg_dir", default=None,
                    help="path to terrain_dreamer_bringup (default: auto)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pkg = Path(args.pkg_dir) if args.pkg_dir else Path(__file__).resolve().parent.parent
    worlds_dir = pkg / "worlds"
    heightmaps_dir = worlds_dir / "heightmaps"
    worlds_dir.mkdir(parents=True, exist_ok=True)

    for env, rocks in ENV_ROCKS.items():
        hm_path = (heightmaps_dir / f"{env}.png").resolve()
        if not hm_path.exists():
            print(f"  [skip] {env} — heightmap missing ({hm_path})")
            continue
        body = HEADER.format(
            env=env,
            heightmap_path=str(hm_path),
            sx=WORLD_SIZE_M, sy=WORLD_SIZE_M, sz=HEIGHT_Z_SCALE_M,
        )
        body += generate_rocks(env, args.seed, **rocks)
        body += FOOTER
        out = worlds_dir / f"{env}.world"
        out.write_text(body)
        print(f"  [ok]   {out.name}  rocks={rocks['n']}")


if __name__ == "__main__":
    main()
