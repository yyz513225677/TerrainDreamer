"""Gz Sim marker manager + traversability mask + terrain sampler.

Used by `ros_jackal_env.py` to spawn the visible start (green) and goal
(red) waypoints in the running gz world, plus to query whether sampled
goals are reachable by the rover (the slope+boulder traversability mask
that `build_varied_world.py` emits alongside the heightmap).

Lessons baked in:
  * NO <transparency> in spawned SDF — triggers OGRE HlmsLowLevel
    "Fixed Function pipeline no longer allowed" exception that stalls
    the gz-sim Sensors plugin and breaks /imu/data flow.
  * NO over-clever per-call materials — all markers reuse the same
    inline materials so OGRE dedupes Hlms datablocks.
  * Spawned via gz transport CLI (`gz service`) because gz.transport13
    Python bindings live in /usr/lib/python3/dist-packages and pulling
    them into the venv would force --system-site-packages.
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# Match build_varied_world.py: 100 m × 100 m world, ±5 m vertical, 513 px.
_WORLD_M = 100.0
_VERT_M  = 10.0


# ── terrain sampler (used for marker z, NOT for env's sensor pipeline) ─────
class TerrainSampler:
    """Read a heightmap PNG once, sample ground z at world (x, y).

    Used so spawned markers sit on the ground instead of buried/floating.
    Falls back to z=0 (the flat world's ground plane) when no heightmap
    is registered for the env_name.
    """
    def __init__(self, png_path: Path):
        from PIL import Image
        img = Image.open(png_path).convert("L")
        self.h_px = np.asarray(img, dtype=np.float32)
        self.res = self.h_px.shape[0]
        self.px = _WORLD_M / (self.res - 1)

    def sample(self, x: float, y: float) -> float:
        col = int(round((x + _WORLD_M / 2) / self.px))
        row = int(round((_WORLD_M / 2 - y) / self.px))
        col = max(0, min(self.res - 1, col))
        row = max(0, min(self.res - 1, row))
        return float(self.h_px[row, col]) / 255.0 * _VERT_M - _VERT_M / 2


def _heightmap_dir() -> Path:
    here = Path(__file__).resolve()
    return (here.parents[3] / "ros_ws/src/terrain_dreamer_bringup"
            / "worlds/heightmaps")


def _find_heightmap(env_name: Optional[str]) -> Optional[Path]:
    if not env_name:
        return None
    candidate = _heightmap_dir() / f"{env_name}.png"
    return candidate if candidate.exists() else None


# ── traversability mask ────────────────────────────────────────────────────
class TraversabilityMask:
    """Loads `<env>_traversable.npy` (boolean RES×RES grid emitted by
    build_varied_world.py) and answers `is_drivable(x_world, y_world)`
    + `sample_drivable_goal(rng, max_dist, origin, min_dist)`."""

    @classmethod
    def find_for(cls, env_name: Optional[str]) -> Optional["TraversabilityMask"]:
        if not env_name:
            return None
        npy = _heightmap_dir() / f"{env_name}_traversable.npy"
        if not npy.exists():
            return None
        try:
            return cls(npy)
        except Exception:
            return None

    def __init__(self, npy_path: Path):
        self.mask = np.load(npy_path).astype(bool)
        self.res = self.mask.shape[0]
        self.px = _WORLD_M / (self.res - 1)

    def is_drivable(self, x: float, y: float) -> bool:
        col = int(round((x + _WORLD_M / 2) / self.px))
        row = int(round((_WORLD_M / 2 - y) / self.px))
        if not (0 <= col < self.res and 0 <= row < self.res):
            return False
        return bool(self.mask[row, col])

    def sample_drivable_goal(
        self,
        rng,                      # np.random.Generator
        max_dist: float,
        origin: tuple = (0.0, 0.0),
        min_dist: float = 0.0,
        max_tries: int = 200,
    ) -> Optional[tuple]:
        ox, oy = origin
        for _ in range(max_tries):
            r = float(rng.uniform(min_dist, max_dist))
            theta = float(rng.uniform(-math.pi, math.pi))
            x = ox + r * math.cos(theta)
            y = oy + r * math.sin(theta)
            if self.is_drivable(x, y):
                return (x, y)
        return None


# ── marker SDF ─────────────────────────────────────────────────────────────
# NO <transparency> — that triggers HlmsLowLevel "Fixed Function pipeline
# no longer allowed" and stalls the gz Sensors plugin. Use bright emissive
# fully-opaque visuals — they're plenty visible from across the world.
_SDF_TEMPLATE = """\
<?xml version="1.0"?>
<sdf version="1.10">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <pose>0 0 {disc_z} 0 0 0</pose>
      <visual name="disc">
        <geometry><cylinder><radius>{disc_r}</radius><length>0.18</length></cylinder></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
          <emissive>{r_em} {g_em} {b_em} 1</emissive>
        </material>
      </visual>
      <visual name="pole">
        <pose>0 0 {pole_z} 0 0 0</pose>
        <geometry><cylinder><radius>0.10</radius><length>{pole_h}</length></cylinder></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
          <emissive>{r_em} {g_em} {b_em} 1</emissive>
        </material>
      </visual>
      <visual name="ball">
        <pose>0 0 {ball_z} 0 0 0</pose>
        <geometry><sphere><radius>0.55</radius></sphere></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
          <emissive>{r} {g} {b} 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def _build_sdf(name: str, color: Tuple[float, float, float],
               radius: float = 1.0) -> str:
    r, g, b = color
    pole_h = 4.5
    return _SDF_TEMPLATE.format(
        name=name, disc_r=radius,
        disc_z=0.10,
        pole_z=pole_h * 0.5 + 0.20,
        pole_h=pole_h,
        ball_z=pole_h + 0.70,
        r=r, g=g, b=b,
        r_em=r * 0.85, g_em=g * 0.85, b_em=b * 0.85,
    )


COLOR_GREEN = (0.10, 0.95, 0.20)
COLOR_RED   = (0.98, 0.10, 0.10)


def _gz_call(service: str, req_type: str, rep_type: str, req: str,
              timeout_ms: int = 3000) -> bool:
    try:
        out = subprocess.run(
            ["gz", "service", "-s", service,
             "--reqtype", req_type, "--reptype", rep_type,
             "--timeout", str(timeout_ms),
             "--req", req],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return "data: true" in out.stdout


class MarkerManager:
    """Spawn / update / delete start+goal markers in the running gz world."""

    def __init__(self, env_name: Optional[str] = None,
                 world_name: str = "moon_flat"):
        self.world = world_name
        self._spawned: set = set()
        env_name = env_name or os.environ.get("TD_ENV_NAME")
        png = _find_heightmap(env_name)
        if png is not None:
            try:
                self._terrain: Optional[TerrainSampler] = TerrainSampler(png)
                print(f"[markers] terrain heightmap loaded: {png.name}")
            except Exception as e:
                print(f"[markers] could not load heightmap {png}: {e}")
                self._terrain = None
        else:
            self._terrain = None

    def _update(self, name: str, x: float, y: float,
                 color: Tuple[float, float, float],
                 radius: float = 1.0):
        if name in self._spawned:
            self._delete(name)
        z = self._terrain.sample(x, y) if self._terrain is not None else 0.0
        sdf = _build_sdf(name, color, radius=radius)
        sdf_esc = sdf.replace('"', '\\"').replace("\n", "\\n")
        req = (
            f'name: "{name}", '
            f'sdf: "{sdf_esc}", '
            f'pose: {{ '
            f'  position: {{ x: {x}, y: {y}, z: {z} }}, '
            f'  orientation: {{ x: 0, y: 0, z: 0, w: 1 }} '
            f'}}'
        )
        if _gz_call(f"/world/{self.world}/create",
                     "gz.msgs.EntityFactory", "gz.msgs.Boolean", req):
            self._spawned.add(name)

    def _delete(self, name: str):
        req = f'name: "{name}", type: 2'   # type 2 = MODEL
        _gz_call(f"/world/{self.world}/remove",
                  "gz.msgs.Entity", "gz.msgs.Boolean", req)
        self._spawned.discard(name)

    def update_start_goal(self,
                          start_xy: Tuple[float, float],
                          goal_xy: Tuple[float, float]):
        self._update("td_start", start_xy[0], start_xy[1], COLOR_GREEN, 0.9)
        self._update("td_goal",  goal_xy[0],  goal_xy[1],  COLOR_RED,   1.2)

    def clear(self):
        for n in list(self._spawned):
            self._delete(n)
