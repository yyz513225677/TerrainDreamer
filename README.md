# TerrainDreamer

World-model-based autonomous navigation for a lunar UGV.
A Clearpath Jackal rover learns to drive to goals `(x, y)` on procedurally
generated Moon terrain using a 32-ring LiDAR + IMU and a DreamerV3-style
RSSM world model. Training is fully online in **Gazebo Classic 11 + ROS 1
Noetic**.

## Architecture

```
Goal (x, y)
     │
     ▼
┌─────────────────────────────────────────────────┐
│                  TerrainDreamer                 │
│                                                 │
│  LiDAR (32-ring)  ──▶ PointCloudProcessor       │
│  IMU   (VN-100)   ──▶ EgoState                  │
│  /ground_truth/odom ─▶ pose (reward + retrace)  │
│                          │                      │
│                          ▼                      │
│              TerrainEncoder (PointPillars)      │
│                          │                      │
│                          ▼                      │
│                  RSSM World Model               │
│                  h_t ──▶ h_{t+1}                │
│                          │                      │
│                          ▼                      │
│         DreamerActor  +  DreamerCritic          │
│      (imagination-trained, goal-conditioned)    │
│                          │                      │
│                          ▼                      │
│              /cmd_vel  (lin_vel, ang_vel)       │
└─────────────────────────────────────────────────┘
```

| Component | Details |
|---|---|
| **Simulator** | Gazebo Classic 11, Moon gravity `-1.62 m/s²`, heightmap terrain + scattered rocks |
| **Robot**     | Clearpath Jackal (stock URDF + VLP-32 + ground-truth p3d) |
| **LiDAR**     | Velodyne HDL-32E stand-in for VLP-32: 32 rings × 512 samples @ 10 Hz, GPU raycast |
| **IMU**       | Jackal UM7 → VN-100 role on `/imu/data` |
| **World model** | RSSM — deter 512 + stoch 64×64 latent state |
| **Encoder**   | PointPillars — `[N, 8]` features → `[256]` embed |
| **Actor**     | Gaussian MLP on `(RSSM state ‖ goal_obs_4d)` → `(lin_vel, ang_vel)` |
| **Critic**    | EMA-target value network for imagination-based TD learning |
| **goal_obs**  | `(dx_norm, dy_norm, dist_norm, heading_err_norm)`, each ∈ [−1, 1] |

---

## Lunar Simulator (Gazebo)

### Environments

| World | Terrain character |
|---|---|
| `mare`          | Smooth volcanic plains — easiest |
| `highland`      | Rugged hills and ridgelines |
| `cratered`      | Impact craters, ejecta rims |
| `boulder_field` | Dense scattered rock clusters |
| `rille`         | Elongated channel cut across the map |

Worlds are procedurally generated:
- **Heightmaps** (513×513 PNG, `[0, 10 m]` Z-scale) via fBm noise + craters + mesas + rilles
- **Rocks** scattered per-env as static spheres with deterministic seeds
- **Lighting** tuned for the Moon (strong directional sun, low ambient)

Regenerate everything:

```bash
python3 ros_ws/src/terrain_dreamer_bringup/scripts/generate_heightmaps.py
python3 ros_ws/src/terrain_dreamer_bringup/scripts/generate_worlds.py
```

### Topics published by `moon_jackal.launch`

| Topic | Type | Source |
|---|---|---|
| `/velodyne_points`   | `sensor_msgs/PointCloud2` | VLP-32 (GPU raycast) |
| `/imu/data`          | `sensor_msgs/Imu`         | Jackal UM7 |
| `/ground_truth/odom` | `nav_msgs/Odometry`       | Gazebo p3d plugin |
| `/jackal_velocity_controller/odom` | `nav_msgs/Odometry` | wheel odom |
| `/cmd_vel`           | `geometry_msgs/Twist`     | **input** (RL action) |

---

## Quickstart

```bash
# 1. Build the ROS workspace (one-time)
cd ros_ws && catkin_make && cd ..

# 2. Python deps
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Generate moon worlds (one-time)
python3 ros_ws/src/terrain_dreamer_bringup/scripts/generate_heightmaps.py
python3 ros_ws/src/terrain_dreamer_bringup/scripts/generate_worlds.py

# 4. Launch autonomous training (headless by default)
./run_auto.sh
./run_auto.sh --gui                     # show Gazebo GUI
./run_auto.sh --env boulder_field       # pick a harder world
./run_auto.sh --resume checkpoints_auto/ckpt_latest.pt
```

`run_auto.sh` handles all ROS env-var plumbing:

- sources `/opt/ros/noetic/setup.bash` and `ros_ws/devel/setup.bash`
- exports `JACKAL_URDF_EXTRAS` and `GAZEBO_MODEL_PATH`
- wraps headless Gazebo in `xvfb-run` so GPU LiDAR raycast has a display
- launches `moon_jackal.launch`, waits for `/ground_truth/odom`, then starts
  `scripts/train_dreamer_auto.py`
- cleans up `gzserver`/`gzclient`/`rosmaster` on exit or Ctrl-C

---

## Autonomous Training Loop

Each mission:

```
1. Sample goal         — annulus [0.6·dmax, dmax] at random bearing from origin
2. GOING phase         — actor drives to goal; heuristic stop-and-turn used during warmup
      • step data (features, action, reward, continue, goal_obs) streamed to replay buffer
      • flip → −50 reward, terminate; timeout → truncate
3. If goal reached:
      a. RETURNING phase — pure-pursuit retraces the GOING waypoints in reverse
      b. Return trajectory also stored (with goal = origin)
4. If goal missed:
      HER relabel — recompute goal_obs + shaping against the final achieved pose
5. World-model update  — RSSM + encoder + reward/continue heads on sampled sequences
6. Actor-critic update — imagination rollouts (DreamerV3 λ-return)
7. Curriculum update   — rolling success rate adjusts dmax:
      > 0.7 → dmax += 1 m   (cap 25 m)
      < 0.3 → dmax -= 1 m   (floor 4 m)
8. Checkpoint every 25 missions → checkpoints_auto/ckpt_mNNNNN.pt
```

### Key flags of `scripts/train_dreamer_auto.py`

| Flag | Default | Description |
|---|---|---|
| `--total_missions` | `500` | Stop after N complete missions |
| `--max_steps`      | `600` | Cap per GOING or RETURNING phase |
| `--warmup_missions`| `10`  | Heuristic-only data collection before engaging the actor |
| `--seq_len`        | `16`  | Sub-sequence length for world-model training |
| `--batch_size`     | `16`  | Model update batch size |
| `--imagine_h`      | `15`  | Imagination rollout horizon |
| `--updates_per_train` | `4` | Gradient updates per mission (when buffer is ready) |
| `--use_actor_prob_end` | `0.85` | Final probability of using the actor vs heuristic |
| `--checkpoint_dir` | `checkpoints_auto` | Output directory |
| `--resume PATH`    | —     | Resume from a `.pt` checkpoint |

### Log output (`checkpoints_auto/training_log.csv`)

Columns: `mission, phase, steps, reward, reached, flipped, goal_dist_max,
success_rate, wm_total, actor_loss, critic_loss, imagined_return`.

---

## Project Structure

```
terrain_dreamer/
├── README.md
├── requirements.txt
├── run_auto.sh                         # ROS + training orchestrator
│
├── ros_ws/                             # catkin workspace
│   └── src/
│       └── terrain_dreamer_bringup/
│           ├── package.xml
│           ├── CMakeLists.txt
│           ├── launch/moon_jackal.launch
│           ├── urdf/jackal_vlp32.urdf.xacro   # VLP-32 + p3d plugin
│           ├── worlds/<env>.world             # generated per env
│           ├── worlds/heightmaps/<env>.png    # generated per env
│           ├── models/                        # custom Gazebo models
│           └── scripts/
│               ├── generate_heightmaps.py
│               └── generate_worlds.py
│
├── src/terrain_dreamer/                # Python package
│   ├── envs/
│   │   ├── ros_jackal_env.py           # Gymnasium wrapper over ROS + Gazebo
│   │   └── sensors/velodyne_vlp32.py   # raw VLP-32C driver + PointCloud dataclass
│   ├── preprocessing/
│   │   ├── point_cloud_processor.py    # raw scan → [N, 8] pillar features
│   │   └── terrain_encoder.py          # PointPillars encoder → 256-d embed
│   ├── world_model/
│   │   ├── rssm.py                     # Recurrent State Space Model
│   │   ├── terrain_dreamer_model.py    # encoder + RSSM + decoder heads
│   │   └── dreamer_policy.py           # DreamerActor, DreamerCritic, imagine_train
│   ├── training/
│   │   └── dreamer_buffer.py           # sequence replay buffer
│   ├── evaluation/ visualization/ policy/ utils/
│
├── scripts/
│   └── train_dreamer_auto.py           # autonomous Dreamer training loop
│
├── checkpoints_auto/                   # checkpoints + training_log.csv + gazebo.log
└── venv/                               # Python virtualenv
```

---

## Environment

- Ubuntu 20.04
- ROS 1 Noetic + Gazebo Classic 11
- Python 3.8 in `venv/`
- CUDA 12.1, PyTorch 2.3.1+cu121
- apt packages: `ros-noetic-jackal-simulator`, `ros-noetic-velodyne-simulator`,
  `ros-noetic-hector-gazebo-plugins`, `xvfb`
