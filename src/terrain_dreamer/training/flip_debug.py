"""
FlipDebugRecorder — per-episode trace + post-hoc PNG dump on flip.

Usage from the training loop::

    rec = FlipDebugRecorder(env_name="mare", out_dir=Path("checkpoints_auto/flip_debug"))
    rec.begin(mission=m, spawn_xy=spawn_xy, goal_xy=goal)
    for t in range(max_steps):
        ...
        obs, r, term, trunc, info = env.step(action)
        rec.step(pose_xy=obs["pose"][:2],
                 pitch_roll=env.latest_pitch_roll(),
                 action=action, reward=r, dist=info["dist_to_goal"])
        if info.get("flipped"):
            rec.dump(reason="flipped")
            break
    rec.clear()  # drop buffers for the next episode if nothing was saved

Produces a 4-panel matplotlib PNG:
  1. Heightmap (gray) + trajectory (color = time) + spawn/goal markers
  2. Pitch, roll, max-tilt vs step number; dashed flip threshold
  3. Commanded linear / angular action vs step number
  4. Distance-to-goal + step reward vs step number

Matplotlib is imported lazily so `import flip_debug` is cheap and the training
script stays usable on headless nodes without mpl installed.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# Heightmaps + world size live next to the .world files. Matches
# ros_ws/src/terrain_dreamer_bringup/worlds/heightmaps/metadata.yaml
_HEIGHTMAP_DIR = (
    Path(__file__).resolve().parents[3]
    / "ros_ws/src/terrain_dreamer_bringup/worlds/heightmaps"
)
_WORLD_SIZE_M = 100.0   # matches metadata.yaml world_size_m
_FLIP_DEG = 60.0        # matches FLIP_PITCH_ROLL in ros_jackal_env


@dataclass
class FlipDebugRecorder:
    env_name: str
    out_dir: Path
    max_cache: int = 10                # keep at most this many PNGs per env
    # filled by begin()/step():
    mission: int = 0
    spawn_xy: Tuple[float, float] = (0.0, 0.0)
    goal_xy:  Tuple[float, float] = (0.0, 0.0)
    path:     List[Tuple[float, float]] = field(default_factory=list)
    pitch_deg: List[float] = field(default_factory=list)
    roll_deg:  List[float] = field(default_factory=list)
    lin_cmd:   List[float] = field(default_factory=list)
    ang_cmd:   List[float] = field(default_factory=list)
    dist:      List[float] = field(default_factory=list)
    reward:    List[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    # capture
    # ------------------------------------------------------------------
    def begin(self, mission: int,
              spawn_xy, goal_xy) -> None:
        self.mission = int(mission)
        self.spawn_xy = (float(spawn_xy[0]), float(spawn_xy[1]))
        self.goal_xy  = (float(goal_xy[0]),  float(goal_xy[1]))
        self.path.clear()
        self.pitch_deg.clear()
        self.roll_deg.clear()
        self.lin_cmd.clear()
        self.ang_cmd.clear()
        self.dist.clear()
        self.reward.clear()

    def step(self, *, pose_xy,
             pitch_roll: Tuple[float, float],
             action,
             reward: float,
             dist: float) -> None:
        self.path.append((float(pose_xy[0]), float(pose_xy[1])))
        self.pitch_deg.append(math.degrees(pitch_roll[0]))
        self.roll_deg .append(math.degrees(pitch_roll[1]))
        self.lin_cmd  .append(float(action[0]))
        self.ang_cmd  .append(float(action[1]))
        self.dist     .append(float(dist))
        self.reward   .append(float(reward))

    def clear(self) -> None:
        self.path.clear()
        self.pitch_deg.clear()
        self.roll_deg.clear()
        self.lin_cmd.clear()
        self.ang_cmd.clear()
        self.dist.clear()
        self.reward.clear()

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------
    def dump(self, *, reason: str = "flipped") -> Optional[Path]:
        """Render the 4-panel PNG. Returns the path, or None if mpl not installed."""
        if not self.path:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[flip_debug] matplotlib unavailable ({e}) — skipping dump")
            return None

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._rotate_cache()

        steps = np.arange(1, len(self.path) + 1)
        px = np.array([p[0] for p in self.path])
        py = np.array([p[1] for p in self.path])
        pitch = np.array(self.pitch_deg)
        roll  = np.array(self.roll_deg)
        tilt  = np.maximum(np.abs(pitch), np.abs(roll))

        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(
            f"env={self.env_name}  mission={self.mission:05d}  reason={reason}  "
            f"steps={len(steps)}  final_tilt={tilt[-1]:.1f}°",
            fontsize=11,
        )

        # --- Panel 1: heightmap + trajectory ---
        ax = axs[0, 0]
        hm = self._load_heightmap()
        if hm is not None:
            ax.imshow(
                hm, cmap="gray", origin="lower",
                extent=(-_WORLD_SIZE_M / 2, _WORLD_SIZE_M / 2,
                        -_WORLD_SIZE_M / 2, _WORLD_SIZE_M / 2),
                alpha=0.8,
            )
        # Trajectory color-coded by step index
        ax.scatter(px, py, c=steps, cmap="plasma", s=14, zorder=3)
        ax.plot([self.spawn_xy[0]], [self.spawn_xy[1]],
                "g^", markersize=12, label="spawn", zorder=4)
        ax.plot([self.goal_xy[0]], [self.goal_xy[1]],
                "r*", markersize=16, label="goal", zorder=4)
        ax.plot([px[-1]], [py[-1]],
                "bx", markersize=14, markeredgewidth=3, label="flip point", zorder=5)
        # Zoom: pad around spawn + goal + path
        xs = np.concatenate([px, [self.spawn_xy[0], self.goal_xy[0]]])
        ys = np.concatenate([py, [self.spawn_xy[1], self.goal_xy[1]]])
        pad = 5.0
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)
        ax.set_aspect("equal")
        ax.set_title("trajectory (color = step)")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

        # --- Panel 2: tilt over time ---
        ax = axs[0, 1]
        ax.plot(steps, pitch, label="pitch", color="tab:blue")
        ax.plot(steps, roll,  label="roll",  color="tab:orange")
        ax.plot(steps, tilt,  label="max|tilt|", color="tab:red",
                linewidth=2, alpha=0.7)
        ax.axhline(+_FLIP_DEG, color="k", linestyle="--", alpha=0.4)
        ax.axhline(-_FLIP_DEG, color="k", linestyle="--", alpha=0.4)
        ax.set_title("pitch / roll / tilt [deg]")
        ax.set_xlabel("step"); ax.set_ylabel("angle [deg]")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)

        # --- Panel 3: commanded action ---
        ax = axs[1, 0]
        ax.plot(steps, self.lin_cmd, label="lin (norm)", color="tab:green")
        ax.plot(steps, self.ang_cmd, label="ang (norm)", color="tab:purple")
        ax.axhline(0, color="k", alpha=0.3)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title("action (normalized to [-1, 1])")
        ax.set_xlabel("step"); ax.set_ylabel("command")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)

        # --- Panel 4: distance + reward ---
        ax = axs[1, 1]
        ax.plot(steps, self.dist, color="tab:cyan", label="dist to goal [m]")
        ax.set_xlabel("step"); ax.set_ylabel("dist [m]", color="tab:cyan")
        ax.tick_params(axis="y", labelcolor="tab:cyan")
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(steps, self.reward, color="tab:red", linestyle=":",
                 label="step reward")
        ax2.set_ylabel("reward", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax.set_title("progress")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = self.out_dir / f"m{self.mission:05d}_{reason}.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _load_heightmap(self) -> Optional[np.ndarray]:
        path = _HEIGHTMAP_DIR / f"{self.env_name}.png"
        if not path.exists():
            return None
        try:
            from PIL import Image
            return np.array(Image.open(path).convert("L"))
        except Exception:
            return None

    def _rotate_cache(self) -> None:
        """Keep only the `max_cache` most recent PNGs per env to bound disk use."""
        pngs = sorted(self.out_dir.glob(f"m*_*.png"))
        excess = len(pngs) - self.max_cache + 1
        for p in pngs[:max(0, excess)]:
            try: os.remove(p)
            except OSError: pass
