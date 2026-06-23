#!/usr/bin/env python3
"""Generate an SDF model directory of procedural rock hazards.

We do not implement a physics or rock engine — each rock is a single
SDF <sphere> or <box> primitive built into one model that Gazebo's
existing engine renders. This file just emits the SDF string.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import yaml


MODEL_CONFIG = """\
<?xml version="1.0"?>
<model>
  <name>lunar_rocks</name>
  <version>1.0</version>
  <sdf version="1.10">model.sdf</sdf>
  <description>Procedurally placed rocks — generated.</description>
</model>
"""


def _rock_xml(idx: int, x: float, y: float, z: float, r: float,
              amb: list[float], diff: list[float]) -> str:
    amb_s = " ".join(f"{c:.3f}" for c in amb)
    diff_s = " ".join(f"{c:.3f}" for c in diff)
    return (
        f"    <link name=\"rock_{idx}\">\n"
        f"      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>\n"
        f"      <collision name=\"col\"><geometry><sphere>"
        f"<radius>{r:.3f}</radius></sphere></geometry></collision>\n"
        f"      <visual name=\"vis\"><geometry><sphere>"
        f"<radius>{r:.3f}</radius></sphere></geometry>\n"
        f"        <material><ambient>{amb_s}</ambient>"
        f"<diffuse>{diff_s}</diffuse></material></visual>\n"
        f"    </link>\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                    help="terrain_params.yaml")
    ap.add_argument("--pkg", required=True, type=Path,
                    help="ROS 2 package directory")
    args = ap.parse_args()

    if not args.config.is_file():
        print(f"[hazards] config not found: {args.config}",
              file=sys.stderr)
        return 2
    if not args.pkg.is_dir():
        print(f"[hazards] package dir not found: {args.pkg}",
              file=sys.stderr)
        return 2

    cfg = yaml.safe_load(args.config.read_text())
    rocks_cfg = cfg.get("rocks", {})
    count = int(rocks_cfg.get("count", 0))
    if count <= 0:
        print("[hazards] count<=0; nothing to do.")
        return 0

    rmin = float(rocks_cfg.get("min_radius_m", 0.05))
    rmax = float(rocks_cfg.get("max_radius_m", 0.40))
    half = float(rocks_cfg.get("half_extent_m", 800.0))
    keepout = float(rocks_cfg.get("keepout_radius_m", 5.0))
    seed = int(rocks_cfg.get("seed", 0))
    amb = list(rocks_cfg.get("ambient", [0.25, 0.22, 0.20, 1.0]))
    diff = list(rocks_cfg.get("diffuse", [0.30, 0.27, 0.24, 1.0]))

    rnd = random.Random(seed)
    rocks_xml = []
    placed = 0
    attempts = 0
    while placed < count and attempts < count * 10:
        attempts += 1
        x = rnd.uniform(-half, half)
        y = rnd.uniform(-half, half)
        if (x * x + y * y) ** 0.5 < keepout:
            continue
        r = rnd.uniform(rmin, rmax)
        # Rest the rock on the ground plane — actual terrain height is
        # unknown here; Gazebo collisions will settle it during sim
        # start. z = r places centre at ground level.
        rocks_xml.append(_rock_xml(placed, x, y, r, r, amb, diff))
        placed += 1

    out_dir = args.pkg / "models" / "lunar_rocks"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.config").write_text(MODEL_CONFIG)
    # static=true so the rocks don't physics-drop at sim start. Their
    # z is set to their radius (centre at ground level); a height-aware
    # placement that samples the DEM PNG is a stage-2 enhancement.
    sdf = (
        '<?xml version="1.0"?>\n'
        '<sdf version="1.10">\n'
        '  <model name="lunar_rocks">\n'
        '    <static>true</static>\n'
        + "".join(rocks_xml)
        + '  </model>\n'
        '</sdf>\n'
    )
    (out_dir / "model.sdf").write_text(sdf)
    print(f"[hazards] wrote {out_dir / 'model.sdf'} "
          f"({placed} rocks placed, seed={seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
