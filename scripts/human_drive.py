"""Human driving + demo recorder for TerrainDreamer (ROS 2 / Gazebo Sim).

Single window combining everything `realtime_viewer.py` showed PLUS the
teleop controls — so `run_human.sh` only needs two visible windows:

    Window 1: Gazebo Sim "Harmonic"    — the UGV in the map
    Window 2: this matplotlib UI       — LiDAR + IMU + trail + controls hint

Keyboard capture uses **pynput** (global, no window focus required), so the
user can keep their cursor anywhere on screen — they don't have to click on
the matplotlib figure to drive.

Recordings are saved as ``demos/demo_NNN.npz`` with the same schema the old
pygame-based ``teleop_record.py`` produced; ``scripts/bc_train.py`` consumes
them unchanged.

Controls (printed in the bottom-left corner of the figure):
    W / S         forward / reverse
    A / D         left / right turn
    SPACE         zero command (emergency stop)
    R             toggle recording
    G             new random goal
    Shift+S       teleport rover to current marker start
    Shift+R       random respawn
    F12           save current demo immediately (also auto-saves on quit)
    Esc / Q       quit
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pynput import keyboard

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from terrain_dreamer.envs.ros_jackal_env import (
    RosJackalEnv, MAX_LINEAR_VEL, MAX_ANGULAR_VEL,
)


# ── demo recorder ──────────────────────────────────────────────────────────

class DemoRecorder:
    def __init__(self):
        self.points:    List[np.ndarray] = []
        self.n_points:  List[int]        = []
        self.imu:       List[np.ndarray] = []
        self.pose:      List[np.ndarray] = []
        self.goal_obs:  List[np.ndarray] = []
        self.action:    List[np.ndarray] = []
        self.spawn_xy:  Optional[np.ndarray] = None
        self.goal_xy:   Optional[np.ndarray] = None

    def add(self, obs: dict, action: np.ndarray):
        self.points  .append(obs["points"].astype(np.float32))
        self.n_points.append(int(obs["n_points"]))
        self.imu     .append(obs["imu"].astype(np.float32))
        self.pose    .append(obs["pose"].astype(np.float32))
        self.goal_obs.append(obs["goal_obs"].astype(np.float32))
        self.action  .append(np.asarray(action, dtype=np.float32))

    def __len__(self):
        return len(self.action)

    def save(self, path: Path) -> bool:
        if len(self) == 0:
            return False
        np.savez_compressed(
            path,
            points=np.stack(self.points),
            n_points=np.asarray(self.n_points, dtype=np.int32),
            imu=np.stack(self.imu),
            pose=np.stack(self.pose),
            goal_obs=np.stack(self.goal_obs),
            action=np.stack(self.action),
            spawn_xy=self.spawn_xy if self.spawn_xy is not None
                      else np.zeros(2, np.float32),
            goal_xy=self.goal_xy if self.goal_xy is not None
                     else np.zeros(2, np.float32),
        )
        return True


def _next_demo_path(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("demo_*.npz"))
    n = 0
    if existing:
        try:
            n = int(existing[-1].stem.split("_")[-1]) + 1
        except ValueError:
            n = len(existing)
    return out_dir / f"demo_{n:03d}.npz"


def _random_goal_drivable(env: RosJackalEnv,
                            np_rng: np.random.Generator,
                            dist: float = 12.0) -> Tuple[float, float]:
    """Sample a goal at ~`dist` metres from origin that lands in a drivable
    cell. Uses the env's traversability mask via env.sample_drivable_goal()."""
    return env.sample_drivable_goal(
        np_rng, max_dist=dist, origin=(0.0, 0.0), min_dist=dist * 0.6,
    )


# ── keyboard listener (pynput) ─────────────────────────────────────────────

class KeyState:
    """Lock-protected keyboard state shared between the listener thread and
    the matplotlib animation thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.held: set = set()           # currently-held key chars (lowercase)
        self.events: List[str] = []      # one-shot events, drained each tick
        self.shift = False
        self.want_quit = False

    def _push_event(self, name: str):
        with self.lock:
            self.events.append(name)

    def drain(self):
        with self.lock:
            ev, self.events = self.events, []
        return ev

    def held_snapshot(self):
        with self.lock:
            return set(self.held), self.shift


_KEY_NAME = {
    keyboard.Key.space: "space",
    keyboard.Key.esc:   "esc",
    keyboard.Key.f12:   "f12",
    keyboard.Key.shift_l: "shift",
    keyboard.Key.shift_r: "shift",
}


def _start_listener(state: KeyState):
    def to_token(key) -> Optional[str]:
        if isinstance(key, keyboard.KeyCode):
            return (key.char or "").lower()
        return _KEY_NAME.get(key)

    def on_press(key):
        tok = to_token(key)
        if tok is None:
            return
        if tok == "shift":
            with state.lock:
                state.shift = True
            return
        with state.lock:
            state.held.add(tok)
        # one-shot events for non-drive keys
        if tok in {"r", "g", "f12", "s"}:
            with state.lock:
                shift = state.shift
            state._push_event(("shift_" if shift else "") + tok)
        if tok == "esc" or tok == "q":
            state.want_quit = True
            return False

    def on_release(key):
        tok = to_token(key)
        if tok is None:
            return
        if tok == "shift":
            with state.lock:
                state.shift = False
            return
        with state.lock:
            state.held.discard(tok)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    return listener


# ── env stepping thread ────────────────────────────────────────────────────

class DriveLoop(threading.Thread):
    """Runs env.step() at step_hz in its own thread so the matplotlib UI
    stays responsive."""

    def __init__(self, env: RosJackalEnv, key_state: KeyState,
                 viz_state: dict, recorder_state: dict,
                 out_dir: Path, np_rng: np.random.Generator,
                 step_hz: float = 10.0):
        super().__init__(daemon=True, name="drive_loop")
        self.env = env
        self.key_state = key_state
        self.viz = viz_state           # shared dict updated in this thread,
                                        # read by matplotlib FuncAnimation
        self.rec_state = recorder_state # holds .recorder, .recording, .demos
        self.out_dir = out_dir
        self.np_rng = np_rng
        self.step_dt = 1.0 / step_hz
        self.running = True

    def stop_recording_and_save(self) -> Optional[str]:
        rec: DemoRecorder = self.rec_state["recorder"]
        if not self.rec_state["recording"]:
            return None
        self.rec_state["recording"] = False
        path = _next_demo_path(self.out_dir)
        if rec.save(path):
            self.rec_state["demos"] += 1
            print(f"[human] saved {path.name}  ({len(rec)} steps)")
            return path.name
        else:
            print("[human] no samples — nothing saved")
            return None

    def start_recording(self):
        rec = DemoRecorder()
        rec.spawn_xy = self.viz.get("spawn_xy", np.zeros(2, np.float32)).copy()
        rec.goal_xy  = self.env._goal.copy()
        self.rec_state["recorder"] = rec
        self.rec_state["recording"] = True
        print(f"[human] REC start  goal={tuple(self.env._goal)}  "
              f"spawn={tuple(rec.spawn_xy)}")

    def _auto_advance(self, dist_to_goal: float):
        """Auto-record + auto-flag pipeline.
        - Outbound (phase=0): when rover reaches goal → save demo, switch
          target to (0,0) (home), start a new recording.
        - Return (phase=1): when rover reaches home → save demo, pick a
          new random outbound goal, start a new recording.

        Guard: the rover must FIRST travel away from the target (dist >
        REACH_FAR) before "reaching" it can fire — otherwise spawning
        right next to a target (e.g. spawn==home on phase 1) creates an
        infinite trigger loop.
        """
        REACH = 0.8
        REACH_FAR = 3.0
        # Track whether the rover has been clearly away from the target
        # since the last phase change.
        if dist_to_goal > REACH_FAR:
            self.viz["left_target"] = True
        if dist_to_goal > REACH or not self.viz.get("left_target", False):
            return
        phase = self.viz.get("mission_phase", 0)
        if phase == 0:   # outbound → return
            self.stop_recording_and_save()
            home = (0.0, 0.0)
            self.env.set_goal(home)
            if self.env._markers is not None:
                self.env._markers.update_start_goal(
                    tuple(self.viz["spawn_xy"]), home)
            self.viz["mission_phase"] = 1
            self.viz["left_target"] = False    # require leaving home next
            print(f"[human] AUTO: reached goal → returning home")
            self.start_recording()
        else:            # return → new outbound
            self.stop_recording_and_save()
            new_goal = _random_goal_drivable(
                self.env, self.np_rng, dist=12.0)
            self.env.set_goal(new_goal)
            if self.env._markers is not None:
                self.env._markers.update_start_goal(
                    tuple(self.viz["spawn_xy"]), new_goal)
            self.viz["mission_phase"] = 0
            self.viz["left_target"] = False    # require leaving spawn next
            print(f"[human] AUTO: reached home → new goal {new_goal}")
            self.start_recording()

    def run(self):
        # Bootstrap: an initial reset so the rover is upright with a drivable
        # goal somewhere ~12 m away.
        initial_goal = _random_goal_drivable(self.env, self.np_rng, dist=12.0)
        obs, info = self.env.reset(options={"goal": initial_goal})
        self.viz["spawn_xy"] = info["spawn_xy"].copy()
        # Auto-start: phase=outbound, recording on immediately.
        self.viz["mission_phase"] = 0
        self.viz["left_target"] = False
        self.start_recording()
        print(f"[human] AUTO mode: goal={initial_goal} recording ON")

        while self.running and not self.key_state.want_quit:
            t0 = time.time()

            # 1) Drain one-shot events.
            for ev in self.key_state.drain():
                if ev == "r":
                    if self.rec_state["recording"]:
                        self.stop_recording_and_save()
                    else:
                        self.start_recording()
                elif ev == "f12":
                    self.stop_recording_and_save()
                elif ev == "g":
                    new_goal = _random_goal_drivable(
                        self.env, self.np_rng, dist=12.0,
                    )
                    self.env.set_goal(new_goal)
                    if self.env._markers is not None:
                        self.env._markers.update_start_goal(
                            tuple(self.viz["spawn_xy"]), new_goal,
                        )
                    if self.rec_state["recording"]:
                        self.rec_state["recorder"].goal_xy = np.array(
                            new_goal, dtype=np.float32,
                        )
                    print(f"[human] new random goal {new_goal}")
                elif ev == "shift_s":
                    sx, sy = self.viz["spawn_xy"]
                    self.env._teleport(float(sx), float(sy), 0.0)
                    self.env._wait_until_settled(timeout=5.0)
                elif ev == "shift_r":
                    obs, info = self.env.reset()
                    self.viz["spawn_xy"] = info["spawn_xy"].copy()
                    if self.rec_state["recording"]:
                        self.stop_recording_and_save()

            # 2) Compute drive command from currently-held keys.
            held, shift = self.key_state.held_snapshot()
            v = 0.0; w = 0.0
            if "w" in held: v += 1.0
            if "s" in held and not shift: v -= 1.0  # Shift+S is teleport, not reverse
            if "a" in held: w += 1.0
            if "d" in held: w -= 1.0
            if "space" in held:
                v = 0.0; w = 0.0
            action = np.array([np.clip(v, -1.0, 1.0),
                               np.clip(w, -1.0, 1.0)], dtype=np.float32)

            obs, _, terminated, truncated, sinfo = self.env.step(action)

            if self.rec_state["recording"]:
                self.rec_state["recorder"].add(obs, action)

            # 3) Publish viz state.
            self.viz["obs"]     = obs
            self.viz["action"]  = action
            self.viz["dist"]    = float(sinfo["dist_to_goal"])

            # 4) Auto-flag swap + auto-record on goal/home touch.
            self._auto_advance(float(sinfo["dist_to_goal"]))

            elapsed = time.time() - t0
            if elapsed < self.step_dt:
                time.sleep(self.step_dt - elapsed)

        # Auto-save on exit.
        if self.rec_state["recording"]:
            self.stop_recording_and_save()


# ── matplotlib UI ──────────────────────────────────────────────────────────

def _make_fig():
    fig = plt.figure(figsize=(11, 6.0), dpi=100)
    fig.patch.set_facecolor("#15151a")
    fig.canvas.manager.set_window_title(
        "TerrainDreamer · human drive  —  LiDAR + IMU + trail"
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.18)
    ax_lid = fig.add_subplot(gs[0])
    ax_imu = fig.add_subplot(gs[1])

    for ax in (ax_lid, ax_imu):
        ax.set_facecolor("#0f0f14")
        for spine in ax.spines.values():
            spine.set_color("#3a3a48")
        ax.tick_params(colors="#bbbbcc")

    ax_lid.set_aspect("equal")
    ax_lid.set_xlim(-12, 12); ax_lid.set_ylim(-8, 16)
    ax_lid.set_xlabel("← left  /  right →", color="#bbbbcc", fontsize=8)
    ax_lid.set_ylabel("↑ forward", color="#bbbbcc", fontsize=8)
    ax_lid.set_title("LiDAR (rover frame, color=height) + path trail",
                      color="#e0e0ee")
    ax_lid.grid(True, color="#26262e", linewidth=0.6)
    import matplotlib.patches as mp
    for r in (3.0, 6.0, 9.0):
        ax_lid.add_patch(mp.Circle((0, 0), r, fill=False,
                                     ec="#3a3a48", lw=0.5, ls="--"))

    ax_imu.set_xlim(-1, 6); ax_imu.set_ylim(-1.0, 1.0)
    ax_imu.set_title("IMU", color="#e0e0ee")
    ax_imu.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax_imu.set_xticks([0, 1, 2, 3, 4, 5])
    ax_imu.set_xticklabels(["pitch", "roll", "yaw", "ax", "ay", "az"],
                            color="#bbbbcc")
    ax_imu.axhline(0, color="#3a3a48", linewidth=0.7)
    return fig, ax_lid, ax_imu


_HINT_TEXT = (
    "[W][A][S][D]  drive          [SPACE] stop\n"
    "[R] toggle record           [G] random goal\n"
    "[Shift+S] teleport home  [Shift+R] respawn\n"
    "[F12] save demo now       [Esc/Q] quit"
)


def _yaw_from_q(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _pitch_roll_from_q(q):
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    return pitch, roll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demos", type=str,
                    help="Directory to write demo_*.npz")
    ap.add_argument("--env-name",
                    default=os.environ.get("TD_ENV_NAME", "varied"))
    ap.add_argument("--step-hz", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    matplotlib.rcParams["toolbar"] = "None"
    print("[human] connecting to ROS 2 / Gazebo Sim …")
    env = RosJackalEnv(step_hz=args.step_hz, env_name=args.env_name)
    env.wait_ready(timeout=20.0)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[human] demos → {out_dir}")

    np_rng = np.random.default_rng(args.seed)
    key_state = KeyState()
    viz: dict = {
        "spawn_xy": np.zeros(2, np.float32),
        "obs": None, "action": np.zeros(2), "dist": 0.0,
        "trail": deque(maxlen=2000),
    }
    rec_state = {"recorder": DemoRecorder(), "recording": False, "demos": 0}

    listener = _start_listener(key_state)

    drive = DriveLoop(env, key_state, viz, rec_state, out_dir, np_rng,
                       step_hz=args.step_hz)
    drive.start()

    # ── matplotlib window ─────────────────────────────────────────────────
    fig, ax_lid, ax_imu = _make_fig()
    scat = ax_lid.scatter([0], [0], s=4, c=[0.0],
                           cmap="viridis", vmin=-1.5, vmax=2.0)
    trail_line, = ax_lid.plot([], [], "-", color="#ffd75f",
                                linewidth=1.6, alpha=0.85, zorder=8)
    rover = ax_lid.scatter([0], [0], s=140, c="#ff5050", marker="^",
                            edgecolors="white", linewidths=1.2, zorder=10)
    heading_line, = ax_lid.plot([0, 0], [0, 3.0], "-",
                                  color="#ff8888", linewidth=1.8, zorder=9)
    fig.colorbar(scat, ax=ax_lid, fraction=0.04, pad=0.02,
                  label="height (m)")

    bars = ax_imu.bar(np.arange(6), np.zeros(6),
                       color=["#5fa8ff"] * 3 + ["#ffa75f"] * 3)

    # Operation hints — bottom-left corner of the figure.
    fig.text(
        0.012, 0.020, _HINT_TEXT,
        color="#dcdcea", fontsize=9, family="monospace",
        va="bottom", ha="left",
        bbox=dict(facecolor="#222230", edgecolor="#3a3a48",
                   boxstyle="round,pad=0.45"),
    )

    # Live status line — top-left of the LiDAR axis.
    txt_status = fig.text(
        0.01, 0.97, "", color="#dddde8",
        fontsize=10, family="monospace", va="top", ha="left",
    )

    # Recording indicator — top-right.
    txt_rec = fig.text(
        0.99, 0.97, "", color="#ff5060",
        fontsize=11, family="monospace", va="top", ha="right",
        bbox=dict(facecolor="#1a1a22", edgecolor="#3a3a48",
                   boxstyle="round,pad=0.40"),
    )

    def update(_frame):
        obs    = viz.get("obs")
        action = viz.get("action", np.zeros(2))
        dist   = viz.get("dist", 0.0)

        if obs is not None and int(obs["n_points"]) > 0:
            n = int(obs["n_points"])
            cloud = obs["points"][:n]
            xs, ys, zs = cloud[:, 0], cloud[:, 1], cloud[:, 2]
            r = np.hypot(xs, ys)
            keep = r > 0.6
            xs, ys, zs = xs[keep], ys[keep], zs[keep]
            plot_x = -ys; plot_y = xs
            scat.set_offsets(np.c_[plot_x, plot_y])
            scat.set_array(np.clip(zs, -1.5, 2.0))

            # Append world pose to trail; transform back into rover frame.
            pose = obs["pose"]   # (x_world, y_world, yaw)
            viz["trail"].append((float(pose[0]), float(pose[1])))
            if len(viz["trail"]) >= 2:
                tw = np.asarray(viz["trail"], dtype=np.float32)
                dx = tw[:, 0] - pose[0]
                dy = tw[:, 1] - pose[1]
                c, s = math.cos(pose[2]), math.sin(pose[2])
                x_rov =  c * dx + s * dy
                y_rov = -s * dx + c * dy
                trail_line.set_data(-y_rov, x_rov)

            # IMU bars
            imu = obs["imu"]   # [wx wy wz ax ay az]
            ax_, ay_, az_ = imu[3], imu[4], imu[5]
            # crude pitch/roll from accel only (no orientation in our IMU msg
            # passthrough — we just visualize gravity tilt on lunar g).
            g = 1.62
            pitch = math.atan2(-ax_, math.sqrt(ay_*ay_ + az_*az_))
            roll  = math.atan2(ay_, az_)
            vals = [
                pitch / (math.pi / 2),
                roll  / (math.pi / 2),
                pose[2] / math.pi,
                np.clip(ax_ / 9.81, -1, 1),
                np.clip(ay_ / 9.81, -1, 1),
                np.clip((az_ - g) / 9.81, -1, 1),
            ]
            for b, v in zip(bars, vals):
                b.set_height(v)
                b.set_color("#ff4060" if abs(v) > 0.6 else
                            ("#5fa8ff" if v >= 0 else "#a85fff"))

            txt_status.set_text(
                f"v={action[0]:+.2f}  ω={action[1]:+.2f}    "
                f"dist={dist:5.2f} m    "
                f"goal=({env._goal[0]:+5.1f},{env._goal[1]:+5.1f})    "
                f"pitch={math.degrees(pitch):+5.1f}° "
                f"roll={math.degrees(roll):+5.1f}°"
            )

        if rec_state["recording"]:
            txt_rec.set_text(
                f"● REC  ({len(rec_state['recorder'])} steps)  "
                f"saved={rec_state['demos']}"
            )
            txt_rec.set_color("#3df062")
        else:
            txt_rec.set_text(f"○ idle    saved={rec_state['demos']}")
            txt_rec.set_color("#888894")

        return scat, trail_line, heading_line, *bars, txt_status, txt_rec

    ani = FuncAnimation(fig, update, interval=80, blit=False,
                         cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        key_state.want_quit = True
        drive.running = False
        drive.join(timeout=2.0)
        listener.stop()
        try:
            env.close()
        except Exception:
            pass
        print(f"[human] done. {rec_state['demos']} demo(s) saved to {out_dir}")


if __name__ == "__main__":
    main()
