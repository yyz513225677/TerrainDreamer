# TerrainDreamer

**Demonstration-Anchored Model-Based RL for Lunar Rover Navigation**

A DreamerV3-based, demonstration-anchored, hierarchical model-based
reinforcement-learning framework for safe Unmanned Ground Vehicle
(UGV) navigation on unstructured, low-visibility off-road terrain.
A Clearpath J100 rover learns to drive to a goal flag `(x, y)` and
return to the spawn point on procedurally-generated lunar
south-pole heightmaps in **Gazebo Sim Harmonic + ROS 2 Jazzy**.

| Configuration | Landscape | Rugged | Extreme |
| --- | --- | --- | --- |
| **TerrainDreamer (full)** | **98.4 ± 1.7 %** | **97.6 ± 2.7 %** | **75.9 ± 4.7 %** |
| Vanilla DreamerV3 (B1) | — | — | 51.4 ± 6.7 % |

5 seeds per terrain, 50 episodes per seed. Δ on extreme vs. B1:
**+24.5 percentage points**.

---

## Contributions

1. **Demonstration-anchored actor loss (C1).** A heavy
   behaviour-cloning term against a programmatic flag-seeking
   demonstrator down-weights the standard imagined-return loss and
   eliminates critic divergence under non-smooth shaping rewards.
2. **Hierarchical sub-goal action space (C2).** The actor emits a
   signed polar sub-goal `(θ, r)` that a fixed P-controller converts
   into a TwistStamped command at 10 Hz. `sign(r)` natively encodes
   reverse manoeuvres.
3. **Triple-source safety filter (C3).** A priority hierarchy of
   interrupts (anti-rollover, stuck recovery, slope detour,
   post-commit, BEV detectors) gated by **multi-source agreement**
   from LiDAR BEV, IMU dynamics, and recent travel-path progress.
   The **progress gate** suppresses every non-safety interrupt while
   the rover is gaining ground on the goal.
4. **Demo distillation (C4).** Whenever a safety interrupt overrides
   the actor, the executed motor command is reverse-mapped back into
   the hierarchical action space and stored in the replay buffer, so
   the world model sees dynamics-consistent transitions and behaviour
   cloning treats interrupt-driven manoeuvres as expert
   demonstrations.

---

## System layout

```
sensors                CRATER core (C1, C2)              filter + distill           robot
─────────              ─────────────────────────         ───────────────            ──────
LiDAR ─┐                Encoder ─▶ RSSM (h_t, s_t) ──┐
IMU   ─┼─▶ encoder ──▶ Actor π_φ   Critic V_ψ        ├─▶ Triple-source ──▶ P-ctrl ──▶ Gazebo
pose  ─┘                Demonstrator π* ─────────────┘   safety filter      (v, ω)     (J100)
                                                         (C3)
                              ▲                                │
                              │      executed (ṽ, ω̃)           │
                              └──── reverse-map ─── buffer ◀───┘
                                       (C4)
```

---

## Repository layout

| Path | Purpose |
| --- | --- |
| `crater/`                       | Core DreamerV3-derived Python package (RSSM, actor / critic, BC heads, safety shield, fusion encoder) |
| `scripts/train_crater_ros.py`   | Single-process ROS 2 + Gazebo + DreamerV3 trainer |
| `scripts/run_seeds.sh`          | 5-seed sweep runner with full per-seed clean restart |
| `scripts/run_ablations.sh`      | B1 / A1 / A2 / A3 ablation runner |
| `scripts/run_paper_experiments_v2.sh` | The full 6-step paper experiment queue |
| `scripts/build_extreme_terrain.py`, `build_rugged_terrain.py`, `build_varied_world.py` | Procedural heightmap generators |
| `lunar_south_pole_gazebo/`      | Gazebo Sim Harmonic world / SDF / launch files + heightmap pipeline |
| `lunar_south_pole_gazebo/ros2_ws/src/lunar_south_pole_gazebo/` | ROS 2 package (mission node, dreamer-interface node, launch) |
| `jackal_dreamer_dashboard/`     | Real-time training dashboard (React + ROS 2 web bridge) |
| `src/terrain_dreamer/`          | Earlier-vintage env + world-model code retained for reference |
| `configs/`, `docs/`             | Configuration files and design notes |

---

## Hardware / software stack

| Layer        | Choice |
| --- | --- |
| OS / kernel  | Ubuntu 22.04 |
| ROS          | ROS 2 Jazzy |
| Simulator    | **Gazebo Sim Harmonic** (gz-sim 8) with `bullet-featherstone` physics at 1 kHz, `real_time_factor = 1.0` |
| Rover model  | Off-the-shelf Clearpath J100 SDF |
| LiDAR        | Velodyne VLP-32 (360°, 32 rings, ±30° vertical, 20 Hz) |
| IMU          | VectorNav VN-100 (100 Hz) |
| Control      | 10 Hz TwistStamped on `/j100_0001/cmd_vel` |
| World model  | DreamerV3 RSSM: 1024-D deterministic + 32×32 categorical stochastic |
| Compute      | Single NVIDIA RTX-class GPU |

The simulator engine and rover model are upstream open-source
artefacts; **we did not modify them**. The contributions of this work
are at the algorithm layer (above) and at the world-content layer
(procedurally-generated heightmaps under `lunar_south_pole_gazebo/`).

---

## Terrains

Three procedurally-generated 100 m × 100 m heightmaps (world
coordinates `x, y ∈ [−50, +50] m`, elevation `z ∈ [0, 4] m`,
513 × 513 pixel grid):

| Terrain | Slope `p_99` | Blocked area | Description |
| --- | --- | --- | --- |
| Landscape | ≈ 5° (median) | ≈ 0 % | Gentle fractal terrain; no craters or boulders |
| Rugged    | 32°           | 7.3 %  | Multi-octave noise + 6 impact craters + river bed + 70 boulders |
| Extreme   | **52.7°**     | **22.0 %** | 8 deep craters with raised rims + eroded trench + 120 large boulders; ±3 m flat clearance only at the spawn point |

---

## Quick start

```bash
# 1. Build the Gazebo workspace once
cd lunar_south_pole_gazebo/ros2_ws
colcon build --packages-select lunar_south_pole_gazebo
source install/setup.bash

# 2. Generate the heightmaps + SDF for a terrain
cd ../..
python3 scripts/build_extreme_terrain.py     # or _rugged_terrain.py
python3 scripts/build_varied_world.py --terrain extreme

# 3. Launch a single training run
TERRAIN=extreme HEADLESS=1 VIEWER=0 \
  python3 scripts/train_crater_ros.py

# 4. Or run the full 5-seed sweep with auto-restart between seeds
TERRAIN=extreme N_SEEDS=5 EPISODES_PER_ITER=50 \
  bash scripts/run_seeds.sh
```

A read-only Gazebo GUI client can be attached at any time without
affecting the headless training server:

```bash
bash scripts/capture_gz_screenshot.sh
```

---

## Ablation protocol

| Tag | Description |
| --- | --- |
| **B1** | Vanilla DreamerV3 — no BC anchor, no hierarchical action, no safety filter |
| **A1** | TerrainDreamer with the triple-source safety filter disabled |
| **A2** | TerrainDreamer with demo distillation disabled (the actor's intended sub-goal is stored verbatim) |
| **A3** | TerrainDreamer with the BC anchor disabled (`λ_BC = 0`) |

Run the full ablation queue with:

```bash
bash scripts/run_paper_experiments_v2.sh   # 6 steps, ~150 h on one GPU
```

The queue runs Landscape and Rugged sweeps under the final algorithm,
then B1, A1, A2, A3 on the extreme terrain. A
`paper_post_ablation_watcher.sh` daemon patches the ablation table
and regenerates the figures every time a step completes.

---

## Reproducibility

The full algorithm, evaluation protocol, hyper-parameters, ROS 2
launch files, and figure-generation scripts are all in this
repository. The paper itself (LaTeX source, figures, BibTeX) is
intentionally excluded from the public repo and kept in a separate
private workspace.

---

## Citation

If TerrainDreamer is useful in your research, please cite (BibTeX
entry will be updated once the paper is officially accepted):

```bibtex
@article{yang2026terraindreamer,
  title   = {{TerrainDreamer}: Demonstration-Anchored Model-Based RL
             for Lunar Rover Navigation},
  author  = {Yang, Yongzhi and Ricks, Kenneth},
  journal = {IEEE Transactions on Intelligent Vehicles (submitted)},
  year    = {2026}
}
```

---

## Acknowledgements

Built on the open-source DreamerV3 reference, Gazebo Sim Harmonic,
ROS 2 Jazzy, and the Clearpath J100 SDF model. Heightmap pipeline
inspired by the OmniLRS lunar simulator.
