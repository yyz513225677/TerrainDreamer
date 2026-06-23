# Jackal-Dreamer Dashboard

A research-lab dashboard for a Jackal UGV running ROS 2 + Gazebo Sim and
controlled by a Dreamer world-model policy. Supports autonomous driving,
human manual control, demonstration recording, route replay, and
**highest-priority** training from recorded expert routes.

![status](https://img.shields.io/badge/MVP-green) ![ros](https://img.shields.io/badge/ROS_2-Jazzy-blue) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

## What's inside

```
jackal_dreamer_dashboard/
├── docs/                  # design + implementation docs
├── dashboard/             # Vite + React + TS + Tailwind dashboard UI
├── ros2_ws/src/           # 5 ROS 2 packages (msgs + control + bridge + recording + safety)
├── data/                  # demonstrations, route manifests, replay buffer
├── tests/                 # mode-FSM, E-STOP, recording-schema tests
└── README.md              # this file
```

The full system architecture is in [`docs/dashboard_design.md`](docs/dashboard_design.md).
The MVP execution plan is in [`docs/implementation_plan.md`](docs/implementation_plan.md).

## Quick start

### 1. Run the dashboard against the **mock bridge** (no ROS required)

Best for UI development + reviews. Two terminals.

```bash
# terminal 1 — start the websocket bridge in mock mode
cd jackal_dreamer_dashboard/ros2_ws
set +u; source /opt/ros/jazzy/setup.bash; source install/setup.bash; set -u
ros2 run jackal_dreamer_bridge dashboard_bridge --mock

# terminal 2 — start the dashboard dev server
cd jackal_dreamer_dashboard/dashboard
npm install     # first time only
npm run dev
# open http://localhost:5173
```

The dashboard also auto-switches to its OWN client-side mock generator if
the bridge isn't reachable, so you can run the frontend completely
standalone with just `npm run dev`.

### 2. Run the dashboard against **live ROS 2 + Gazebo**

Three terminals — assumes a Jackal sim is already publishing /scan,
/imu/data, /ground_truth/odom (any way; the project's own `run.sh` works).

```bash
# terminal 1 — start the Jackal sim (existing terrain_dreamer launcher)
cd /home/rickslab3/Documents/Leo/terrain_dreamer
./run.sh --mode auto         # or use your own launch

# terminal 2 — start the dashboard backend
cd jackal_dreamer_dashboard/ros2_ws
set +u; source /opt/ros/jazzy/setup.bash; source install/setup.bash; set -u
ros2 launch_helpers/run_all.sh        # (optional helper; or run each node directly)
ros2 run jackal_dreamer_bridge   dashboard_bridge
ros2 run jackal_dreamer_control  mode_manager
ros2 run jackal_dreamer_control  manual_control
ros2 run jackal_dreamer_safety   safety_supervisor
ros2 run jackal_dreamer_recording route_recording
ros2 run jackal_dreamer_recording route_replay

# terminal 3 — dashboard
cd jackal_dreamer_dashboard/dashboard
npm run dev
# open http://localhost:5173
```

A single-shot launcher (`scripts/start_all.sh`) is in the work queue —
for now run each node in its own terminal so you can read its log.

### 3. Build the ROS 2 packages from scratch

```bash
cd jackal_dreamer_dashboard/ros2_ws
unset CMAKE_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION VIRTUAL_ENV
set +u; source /opt/ros/jazzy/setup.bash; set -u
colcon build
source install/setup.bash
ros2 pkg list | grep jackal_dreamer
# → jackal_dreamer_bridge
#   jackal_dreamer_control
#   jackal_dreamer_dashboard_msgs
#   jackal_dreamer_recording
#   jackal_dreamer_safety
```

The bridge needs the `websockets` Python package; install once into both
the system Python and the project venv so the entry point runs in either:

```bash
pip install --break-system-packages websockets   # for system python3
/home/rickslab3/Documents/Leo/terrain_dreamer/venv/bin/pip install websockets
```

### 4. Build the dashboard frontend for production

```bash
cd jackal_dreamer_dashboard/dashboard
npm install
npm run build           # → dist/index.html + dist/assets/{index-*.css, index-*.js}
# bundle ≈ 185 KB JS / 14 KB CSS  /  gzipped: 58 KB JS / 3.6 KB CSS
```

Drop `dist/` behind any static server — nginx, `python3 -m http.server`,
or `npx serve dist`.

## Modes and `/cmd_vel` ownership

Only one source publishes to `/cmd_vel` at a time. The `mode_manager_node`
arbitrates; the `safety_supervisor_node` filters everything as a
last-line-of-defense:

| Mode         | Source of `/cmd_vel`                |
|--------------|------------------------------------|
| **Manual**   | `manual_control_node` (W/A/S/D)    |
| **Autonomous** | `dreamer_policy_node` (existing) |
| **Recording**| `manual_control_node` + recorder   |
| **Replay**   | `route_replay_node`                |
| **Training** | none — sim is the training env     |
| **E-STOP**   | `safety_supervisor_node` (zero)    |

E-STOP latches. Press the red button in the top bar to engage; click
`RESET` in the SimControl panel to clear it.

## Human demonstration → priority-1 training

In **Recording Mode**, drive the rover with W/A/S/D and click `START` in
the Recording panel. Every timestep, the system saves a sample matching
[`RouteSample`](ros2_ws/src/jackal_dreamer_dashboard_msgs/msg/RouteSample.msg):

```
data/demonstrations/<route_id>.jsonl          # one JSON object per timestep
data/demonstrations/<route_id>_lidar/*.npy    # per-frame LiDAR ranges
data/routes/<route_id>.json                   # one-shot route manifest
data/replay_buffer/index.json                 # appends with priority=1.0
```

The Dreamer trainer (`scripts/train_dreamer_auto.py` in the parent repo)
reads `data/replay_buffer/index.json` and uses **priority-weighted
sampling**: entries with `priority=1.0` are guaranteed to be sampled at
least once per epoch. That is what makes recorded human routes the
**highest-priority learning path**.

## Tests

```bash
cd jackal_dreamer_dashboard
python3 -m pytest tests/ -v
```

19 tests cover:

* **`test_mode_fsm.py`** — valid transitions, invalid mode rejection,
  E-STOP latching, reset semantics, sequence monotonicity.
* **`test_estop.py`** — E-STOP filters every non-zero Twist; deassert via
  `estop=False` does NOT clear the latch (only `reset` does); rising-edge
  is immediate.
* **`test_recording_schema.py`** — every JSONL line conforms to the
  `RouteSample` schema; nested shapes (`pose`, `imu`, `action`) are
  correct; route manifest matches sample count; expert demos have
  `priority=1.0`; replay buffer contains the seed route.

## Visual design language

Dark professional aerospace / lunar-research-lab. See
[`docs/dashboard_design.md` §4](docs/dashboard_design.md#4-frontend-visual-system).

| Color           | Hex       | Use                              |
|-----------------|-----------|----------------------------------|
| background      | `#0b0d10` | dashboard root                   |
| panel           | `#13171c` | panel fill                       |
| LiDAR cyan      | `#22d3ee` | scan points                      |
| Dreamer blue    | `#3b82f6` | autonomous route                 |
| expert amber    | `#f59e0b` | human demonstration route        |
| danger red      | `#ef4444` | E-STOP, collision markers        |
| nominal green   | `#10b981` | OK status pills                  |

Typography: **JetBrains Mono** with `tabular-nums` for telemetry,
**Inter** uppercase letter-spaced for panel headers, 12-column grid with
16 px gutter, fixed 1440 × 900 baseline.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| dashboard shows "MOCK MODE" banner | bridge isn't reachable at ws://localhost:8765 | start `ros2 run jackal_dreamer_bridge dashboard_bridge`, OR confirm port 8765 isn't firewalled |
| `colcon build` fails with `catkin_pkg` missing | venv-Python shadowed system Python | `unset VIRTUAL_ENV` before `colcon build` |
| `OpenGL 3.3 not supported` from gz sim | NVIDIA driver / kernel module mismatch | reboot after driver upgrade — see parent project's notes |
| websockets `ImportError` from bridge node | `websockets` not in active Python's site-packages | `pip install --break-system-packages websockets` (system) and `venv/bin/pip install websockets` |
| `npm install` times out behind a proxy | corporate NPM mirror | `npm config set registry https://your.mirror/` then retry |
| keyboard control unresponsive | mode is not Manual or Recording | switch mode in the ManualControl panel; check the top-bar mode pill |
| recorded JSONL has 0 frames | recording started but no sensor data arrived | confirm `/imu/data` + `/ground_truth/odom` are publishing on the live bridge |

## Status

* ✅ MVP complete — dashboard + bridge + all 9 modules + tests + verification
* ⏳ Phase 2 — wire to live Gazebo Sim end-to-end (waiting on parent
  repo's sim stack stabilization)
* ⏳ Phase 3 — Dreamer training sampler reads `data/replay_buffer/`
  with priority-weighted importance sampling
* ⏳ Phase 4 — route comparison (Fréchet distance, success rate vs
  reference route)

## License

MIT — see parent repository for full text.
