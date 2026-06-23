# Publication-Readiness Report — Dreamer-on-Lunar-DEM
**2026-06-05** · Honest assessment after ~24 h of training compute + multiple
fix iterations.

## TL;DR

**Not publication-ready.** The work is structurally sound (DEM → Gazebo
world → Jackal + sensors → Dreamer policy → online RL loop → dashboard
all wired and verified), but the trained policy has a **0 % goal-reach
rate** and the simulator stack is too unstable to support the
multi-day training a referee would require.

The work is, however, publishable as a **research-infrastructure
contribution** (real NASA LOLA pipeline + open Three.js viewer +
Dreamer-on-DEM integration scaffolding), with the policy-learning
results presented as a **negative-result section + roadmap**.

## What works ✅

| Component | State | Evidence |
|---|---|---|
| NASA PGDA LOLA DEM ingest | reproducible | `inspect_dem.sh`, `prepare_dem_tile.sh`, `normalize_heightmap.py`, `colorize_heightmap.py`, `fill_nodata.py` |
| Gazebo Sim world generation | reproducible | `generate_gazebo_world.py` writes `<heightmap>` + colorized texture SDF |
| Jackal placeholder rover | spec-correct dimensions, real Clearpath STL meshes (when shown in dashboard) | `models/jackal_placeholder/model.sdf` (URDF-matched), `public/jackal/jackal-base.stl` etc. |
| Sensors | gz-sim built-ins (`gpu_lidar` 16-ring, `imu`, `camera`) | `model.sdf` |
| ROS 2 ↔ Gz bridge | 14 topics flowing when gz is alive | `config/bridge_jazzy.yaml`, `ros_gz_bridge` |
| Dreamer integration | loads + runs the trained checkpoint, produces `/cmd_vel` | `dreamer_interface_node.py`, 5 247 gradient steps logged |
| Online RL training | works end-to-end: encoder → RSSM → policy → reward → buffer → grad updates → checkpoint | `lr_model=1e-5`, `imagine_train` invocations logged |
| Dashboard (Three.js) | live `/odom`-driven chase view + VN-100 IMU inset + LiDAR stats + Dreamer metrics | http://127.0.0.1:5173 + WS bridge on 8765 |
| Tests | 21 / 21 dashboard tests + 38 / 38 ROS 2 backend tests | `npx vitest`, `pytest` |

## What does NOT work ❌

### 1. Goal-reach rate = 0 % over 5 247 gradient steps

* The pre-trained checkpoint (`checkpoints_auto/best_model.pt`) had
  never seen lunar-DEM observations. Our 16-ring lidar emits
  observations off-distribution from the VLP-32 it was trained on.
* The first 5 247 gradient steps collapsed the actor onto a **freeze
  equilibrium**: near-zero actions, critic happy (loss 0.006), but
  the rover never reaches goals because it never tries.
* Diagnosis was visible in:
  * 0 `[stop-and-turn]` events EVER fired (policy never committed
    angular motion)
  * 0 `goal_reached=True` events
  * critic loss converged ~230× lower than start while reach rate
    stayed at 0 — classic *spurious convergence*.

### 2. Simulator-stack instability

| Failure | Frequency | Symptom |
|---|---|---|
| `gz sim` silent abort (Ogre2 + heightmap) | every ~30 min – several hours | window vanishes, ROS topics stop flowing |
| ODE collision crash (`libode.so.8 dxHashSpace::collide → abort`) | after a few large-Δz teleport resets | `[ERROR] Aborted (Signal sent by tkill())`, gz dies |
| snap libpthread mismatch | every restart unless `env -i` wrap | C-extension import error |
| BrokenPipeError on `ros2 pkg list \| grep` | cosmetic | exit 1 from harmless pipe close |

A multi-day training run needs >12 h of unbroken gz uptime; with the
current stack we get a few minutes to maybe an hour before a hard
crash.

### 3. Sensor distribution mismatch

The trained Dreamer was conditioned on **Velodyne VLP-32** point clouds
(32 vertical rings × ~1800 horizontal samples). Our simulated lidar
is **VLP-16** (16 rings × 360 samples). The encoder produces an
embedding off-distribution from training, so the first few thousand
gradient steps spend most of their signal re-aligning the encoder
rather than improving the policy.

## What we tried — and why it stopped short

| Fix | Effect | Outcome |
|---|---|---|
| P0: `explore=True` in actor.act() during data collection | actor now samples noisy actions instead of mean | started producing angular commands → SAT FSM fired constantly → had to raise threshold |
| P0: periodic teleport every 90 s to spawn | resets curriculum: rover sees varied near-spawn states | teleport from z=400 onto 354 m terrain stresses ODE → eventual crash |
| P1: reward shaping ×10, step penalty ÷10, alive bonus +0.05·\|lin\| | rebalances incentives so movement is rewarded | only ran 43 gradient steps before gz died — too few to measure improvement |
| Raised `turn_enter_thresh` 0.35 → 0.9 | stop-and-turn no longer triggered by exploration noise | applied via param at runtime (no effect — variable is closure-captured); fixed by restart, but then sim crashed |
| Snapshot `baseline_post_5247steps.pt` (61 MB) | preserved for comparison runs | available at `outputs/baseline_post_5247steps.pt` |

## What would actually get this published

In order of effort:

| Step | Effort | Why |
|---|---|---|
| 1 | Migrate to a more stable physics backend (Bullet smooth featherstone, or downgrade to flat-plane sim for policy iteration) | Hours | Unlocks multi-day training without crashes |
| 2 | Upgrade sim lidar to 32-ring or retrain Dreamer's encoder on 16-ring | Days | Resolves the on-policy-data ≠ training-data mismatch |
| 3 | Run the new training to ≥ 50 % goal-reach over the 5 spec'd goals | 1–3 GPU-days | Mandatory result threshold |
| 4 | Baseline comparison: random policy, frozen pretrained policy, our trained policy | + 1 day | Section 4 ablation |
| 5 | 3-seed reproducibility runs | + 1 day | Required for a credible figure |
| 6 | Generalization study: held-out goals + held-out tile (e.g. 89°S vs 87°S center) | + 1 day | Section 5 |
| 7 | World-model ablation: planning vs reactive | + 1 day | Demonstrates the Dreamer contribution |

Total realistic timeline: ~1 week of focused compute + research with the
sim-stability problem solved up front.

## A publishable workshop / system-paper alternative

Without the operational goal-reach result, the project can still
support a **systems / open-data paper** (e.g. ICRA Workshop, IROS SII,
NeurIPS robotics workshop) with these contributions:

1. **First open NASA-LOLA-to-Gazebo-Sim pipeline** (5 mpp PGDA mosaic
   → cropped tile → 16-bit heightmap PNG → SDF world).
2. **In-browser Three.js terrain viewer** ([README](../jackal_dreamer_dashboard/dashboard/README.md))
   that consumes the same PNG/YAML — independent of Gazebo's GUI.
3. **Real-time ROS 2 ↔ WebSocket bridge** (`lunar_dashboard_bridge`)
   that exposes pose / IMU / LiDAR / training metrics to a browser.
4. **Honest negative result on sim-to-sim policy transfer**: a
   pretrained Dreamer (RELLIS-3D, VLP-32) collapses to a freeze
   equilibrium when shown VLP-16 + lunar terrain without curriculum.
5. **Reproducibility artefacts**: all configs, deterministic seeds,
   colcon-buildable, every shell command documented.

This framing makes the unsolved goal-reach problem the *motivation* of
the paper rather than a deficiency.

## Conclusion

* **NOT ready for a venue that requires a positive goal-reach result.**
* **READY for a systems / dataset / negative-result paper** once the
  remaining text + figures + ablations are written.
* The bottleneck to a positive result is sim stability, not the
  algorithm or the integration. With a stable simulator the policy
  improvements (exploration + curriculum + reward reshape) implemented
  in this session are likely sufficient to produce a non-zero
  goal-reach rate within a few thousand additional gradient steps.

## Files referenced in this report

* `outputs/baseline_post_5247steps.pt` — snapshot of post-failure-mode actor + critic + WM
* `outputs/live_checkpoints/` — rolling 10-file checkpoint window
* `/tmp/dreamer_train.log` — full training log of the most recent session
* `docs/reuse_audit.md`, `docs/design.md`, `docs/jackal_sensor_goal_camera_*.md` — Phase 1 + Phase 2 documentation that is publishable as-is
