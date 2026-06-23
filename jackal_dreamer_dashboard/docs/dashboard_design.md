# Jackal-Dreamer Dashboard — Design Document

## 1. Purpose

A research-lab dashboard for a Jackal UGV running ROS 2 + Gazebo Sim and
controlled by a Dreamer world-model policy. Operators must be able to:

1. Monitor live robot telemetry (pose, IMU, LiDAR, odometry).
2. Switch the rover between **Manual / Autonomous / Recording / Replay /
   Training** modes safely.
3. Record human demonstrations and feed them into the Dreamer training
   buffer as **highest-priority** expert routes.
4. Replay recorded routes and visually compare them against the
   Dreamer-driven trajectory.
5. Issue an **emergency stop** that overrides every other source of
   `/cmd_vel` at any time.

The dashboard is the **only operator-facing surface** for the project's
human-in-the-loop research workflow. Reliability + clear visual hierarchy
take priority over feature breadth.

## 2. Constraints / non-goals

- **Not** a generic ROS GUI. We expose only the topics + actions our
  project needs.
- **Not** a 3-D viewer — RViz handles that out-of-band. The dashboard's
  visualization is 2-D BEV (bird's-eye view), which is the right
  abstraction for navigation decisions.
- **Not** a video stream. Camera-frame rendering would dominate the
  bandwidth; we send numeric telemetry + decimated LiDAR.

## 3. System architecture

```
 ┌───────────────────────────────────┐        ┌──────────────────────────┐
 │  React + TypeScript dashboard      │ ws://  │  dashboard_bridge_node   │
 │  (Vite dev server, port 5173)      │◄──────►│  websocket on port 8765  │
 └────────────┬──────────────────────┘        │                          │
              │ keyboard / button input        │  publishes:              │
              │ mode + e-stop commands         │   /dashboard/mode        │
              ▼                                │   /dashboard/estop       │
 ┌───────────────────────────────────┐         │   /dashboard/reset       │
 │  manual_control_node               │◄────────┤   /dashboard/recording/* │
 │  (W/A/S/D → /cmd_vel WHEN owned)   │         │   /dashboard/replay/*    │
 └───────────────────────────────────┘         │                          │
                                                │  subscribes:             │
 ┌───────────────────────────────────┐         │   /scan, /imu, /odom,    │
 │  dreamer_policy_node (existing)    │         │   /tf, /clock,           │
 │  (/cmd_vel when Autonomous mode    │         │   /dreamer/metrics,      │
 │   owns control)                    │         │   /collision_status      │
 └───────────────────────────────────┘         └──────────────────────────┘
              ▲                                            ▲
              │                  ┌─────────────────────────┘
              │                  │
 ┌────────────┴──────────────────┴──┐    ┌────────────────────────────────┐
 │   safety_supervisor_node          │    │  route_recording_node          │
 │   - latches E-STOP                │    │  - logs synchronized samples   │
 │   - zeroes /cmd_vel when armed    │    │    while in Recording Mode     │
 │   - requires explicit reset       │    │  - writes JSON manifest +      │
 │   - last-line-of-defense filter   │    │    optional binary LiDAR refs  │
 │     on the cmd_vel topic          │    │  - flags priority=1.0 demos    │
 └───────────────────────────────────┘    └────────────────────────────────┘

                  ┌──────────────────────────────┐
                  │  route_replay_node            │
                  │  - replays saved trajectories │
                  │  - publishes /cmd_vel only if │
                  │    Replay Mode owns control   │
                  │  - publishes pose targets to  │
                  │    /dashboard/replay/pose     │
                  └──────────────────────────────┘
```

### 3.1 Mode ownership of `/cmd_vel`

A single `mode_manager_node` arbitrates ownership. Only one publisher
relays to `/cmd_vel` at a time:

| Mode         | Owner of `/cmd_vel`                |
|--------------|------------------------------------|
| Manual       | `manual_control_node`              |
| Autonomous   | `dreamer_policy_node`              |
| Recording    | `manual_control_node` + recorder   |
| Replay       | `route_replay_node`                |
| Training     | none — sim is the training env     |
| E-STOP       | `safety_supervisor_node` (zero)    |

The safety supervisor latches E-STOP. While latched, every incoming
`/cmd_vel` is overwritten with a zero Twist before reaching the rover.
A separate `/dashboard/reset` action is required to unlatch.

### 3.2 Data flow for human demonstrations

Recording Mode captures one **sample** per timestep at the LiDAR's 10 Hz
rate:

```json
{
  "timestamp": 1759412.413,
  "route_id":     "demo_2026-06-03T18-22-15",
  "operator_id":  "leo",
  "environment_id": "varied",
  "pose":         {"x": 12.34, "y": -3.21, "yaw": 1.5708},
  "odom":         {"linear_x": 0.42, "angular_z": -0.12},
  "imu": {
    "roll": 0.01, "pitch": -0.02, "yaw": 1.5712,
    "angular_velocity": {"x": 0.00, "y": 0.01, "z": -0.12},
    "linear_acceleration": {"x": 0.02, "y": 0.00, "z": -1.62}
  },
  "lidar_ref":    "data/demonstrations/demo_2026-...lidar/000123.npy",
  "action":       {"linear_x": 0.42, "angular_z": -0.12},
  "collision":    false,
  "reward_est":   0.18,
  "priority":     1.0
}
```

A route's metadata (1 file per route) lives in `data/routes/`:

```json
{
  "route_id":     "demo_2026-06-03T18-22-15",
  "operator_id":  "leo",
  "environment_id": "varied",
  "started_at":   "2026-06-03T18:22:15Z",
  "duration_s":   88.4,
  "frame_count":  884,
  "spawn_xy":     [0.0, 0.0],
  "goal_xy":      [38.0, -12.0],
  "outcome":      "reached_goal",
  "collisions":   0,
  "priority":     1.0,
  "samples_path": "data/demonstrations/demo_2026-06-03T18-22-15.jsonl",
  "lidar_dir":    "data/demonstrations/demo_2026-06-03T18-22-15_lidar"
}
```

The replay buffer (`data/replay_buffer/index.json`) keeps a flat list of
route_ids with priority weights. Dreamer's training sampler reads this
list and uses **priority-weighted importance sampling**; routes with
`priority=1.0` (human expert) are guaranteed to be sampled at least once
per epoch.

## 4. Frontend: visual system

Aesthetic: **dark professional aerospace / lunar-research-lab**. The
visual language is borrowed from instrument panels (Saab Gripen,
Apollo-era flight directors), not from web/SaaS design.

### 4.1 Color tokens

| Token              | Hex       | Use                                       |
|--------------------|-----------|-------------------------------------------|
| `bg.base`          | `#0b0d10` | dashboard background                       |
| `bg.panel`         | `#13171c` | panel fill                                 |
| `bg.panel-raised`  | `#1a1f25` | nested raised regions                      |
| `border.subtle`    | `#252b33` | thin technical borders                     |
| `border.strong`    | `#3a4350` | section dividers                            |
| `text.primary`     | `#e3e8ee` | telemetry text                              |
| `text.muted`       | `#7a8590` | labels, secondary info                      |
| `accent.lidar`     | `#22d3ee` | LiDAR points, scan rings                    |
| `accent.dreamer`   | `#3b82f6` | autonomous route, Dreamer outputs           |
| `accent.expert`    | `#f59e0b` | human expert routes                         |
| `accent.warning`   | `#fbbf24` | non-critical alerts                         |
| `accent.danger`    | `#ef4444` | collision marker, E-STOP latched            |
| `accent.ok`        | `#10b981` | nominal status                              |

### 4.2 Typography

| Use                       | Family             | Size  | Weight |
|---------------------------|--------------------|-------|--------|
| telemetry numerals        | JetBrains Mono     | 14 px | 500    |
| panel headers             | Inter (uppercase)  | 11 px | 600 letter-spaced |
| body labels               | Inter              | 12 px | 400    |
| status pills              | JetBrains Mono     | 11 px | 600    |

All numeric readouts use **tabular-nums** so digits don't jitter while
they update.

### 4.3 Layout

12-column grid, 16 px gutter. Default breakpoint: 1440 × 900 (lab
workstations). The layout is **fixed** — no responsive collapse below
1280 px, the dashboard expects a wide monitor.

```
┌────────────────────────────────────────────────────────────────────────┐
│  TOP SYSTEM BAR · ros ● gazebo ● mode · sim-time · health · E-STOP    │
├──────────────────────────┬───────────────────────────┬─────────────────┤
│  ROBOT TELEMETRY         │  MAIN VISUALIZATION        │  DREAMER POLICY │
│  (col 1-3)               │  (col 4-9)                 │  (col 10-12)    │
│  position / vel /        │  2-D BEV map with:         │  v, ω readouts  │
│  IMU bars / collision    │  · terrain hint            │  reward / value │
│  / stuck risk            │  · LiDAR cyan scatter      │  losses (wm,    │
│                          │  · current pose            │  actor, critic, │
│                          │  · amber expert route      │  BC)            │
│                          │  · cyan/blue Dreamer route │  buffer size    │
│                          │  · red collision markers   │  human ratio    │
│                          │  · goal flag               │                 │
├──────────────────────────┴───────────────────────────┼─────────────────┤
│  SIM CONTROL · MANUAL CONTROL · RECORDING · ROUTE LIBRARY              │
│  (col 1-12, 4 horizontal panels of mode + actions)                     │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Component inventory

| Component          | File                                | Tier      |
|--------------------|-------------------------------------|-----------|
| `TopBar`           | `components/TopBar.tsx`             | shell     |
| `EmergencyStop`    | `components/EmergencyStop.tsx`      | shell     |
| `TelemetryPanel`   | `components/TelemetryPanel.tsx`     | panel     |
| `BEVMap`           | `components/BEVMap.tsx`             | viz       |
| `DreamerPanel`     | `components/DreamerPanel.tsx`       | panel     |
| `SimControlPanel`  | `components/SimControlPanel.tsx`    | panel     |
| `ManualControl`    | `components/ManualControl.tsx`      | panel     |
| `RecordingPanel`   | `components/RecordingPanel.tsx`     | panel     |
| `RouteLibrary`     | `components/RouteLibrary.tsx`       | panel     |
| `StatusPill`       | `components/StatusPill.tsx`         | primitive |
| `MetricRow`        | `components/MetricRow.tsx`          | primitive |
| `Panel`            | `components/Panel.tsx`              | primitive |
| `Sparkline`        | `components/Sparkline.tsx`          | primitive |

### 4.5 Data integration

The frontend talks to `dashboard_bridge_node` over a single websocket
(`ws://<host>:8765`). Messages are line-delimited JSON; the bridge
multiplexes ROS topics into the stream and demuxes outgoing commands.

```ts
// Inbound from bridge
type WSInbound =
  | { topic: "telemetry"; payload: TelemetrySample }
  | { topic: "lidar";     payload: { ranges: number[]; intensities?: number[] } }
  | { topic: "dreamer";   payload: DreamerMetrics }
  | { topic: "mode";      payload: { mode: Mode; estop: boolean } }
  | { topic: "route";     payload: RouteEvent };

// Outbound to bridge
type WSOutbound =
  | { cmd: "set_mode";    mode: Mode }
  | { cmd: "estop";       active: boolean }
  | { cmd: "reset" }
  | { cmd: "manual_cmd";  linear: number; angular: number }
  | { cmd: "recording";   action: "start" | "stop" | "save"; route_id?: string;
                          operator_id?: string; environment_id?: string }
  | { cmd: "replay";      action: "start" | "stop"; route_id: string };
```

When the websocket can't reach a bridge, the dashboard auto-switches to
**mock mode**, generating synthetic telemetry so the UI can be developed
and reviewed without a live sim.

## 5. ROS 2 layer

### 5.1 Nodes

| Node                     | Pkg                              | Responsibility |
|--------------------------|----------------------------------|----------------|
| `mode_manager_node`      | `jackal_dreamer_control`         | own current mode, latch E-STOP, gate cmd_vel publishers |
| `dashboard_bridge_node`  | `jackal_dreamer_bridge`          | websocket ↔ ROS multiplexer, downsamples /scan to 360 ranges |
| `manual_control_node`    | `jackal_dreamer_control`         | converts dashboard manual_cmd → /cmd_vel when Manual/Recording owns control |
| `route_recording_node`   | `jackal_dreamer_recording`       | logs synchronized samples to JSONL, writes route manifest, registers replay buffer entry |
| `route_replay_node`      | `jackal_dreamer_recording`       | replays a JSONL route through /cmd_vel; publishes target pose on /dashboard/replay/pose |
| `safety_supervisor_node` | `jackal_dreamer_safety`          | filters /cmd_vel; while E-STOP latched, publishes zero Twist at high rate |

### 5.2 Custom msgs (`jackal_dreamer_dashboard_msgs`)

| Msg                  | Purpose |
|----------------------|---------|
| `DashboardMode.msg`  | string `mode` enum + bool `estop_active` + uint32 `seq` |
| `DreamerMetrics.msg` | float32 wm/actor/critic/bc losses + buffer size + human_ratio + value/reward estimates |
| `CollisionStatus.msg`| bool in_contact + float32 stuck_score + uint32 collision_count |
| `RouteSample.msg`    | full per-timestep recording schema (mirrors the JSONL format) |

### 5.3 Topic catalogue

Subscribes (bridge): `/scan`, `/imu/data`, `/ground_truth/odom`, `/tf`,
`/clock`, `/dreamer/metrics`, `/dreamer/action`, `/collision_status`.

Publishes (bridge / control nodes):
`/cmd_vel`, `/dashboard/mode`, `/dashboard/estop`, `/dashboard/reset`,
`/dashboard/recording/{start,stop,save}`, `/dashboard/replay/{start,stop}`,
`/dashboard/training/start`.

## 6. Safety design

- E-STOP is **client-side instantaneous** (UI publishes immediately on
  click) **AND server-side latched** (`safety_supervisor_node` keeps
  zeroing `/cmd_vel` until it sees an explicit `/dashboard/reset`).
- Manual-control publish rate caps at 20 Hz; idle keys publish zero
  rather than no message (so a dead UI doesn't leave a stale Twist).
- The bridge auto-publishes a zero Twist on websocket disconnect during
  Manual or Recording mode (deadman).
- Mode transitions clear the action history of the previous owner so
  there's no "leftover" Twist when ownership flips.

## 7. Testing strategy

- **Mode switching**: pytest covers the `mode_manager_node` state
  machine — all valid transitions + every E-STOP latch/unlatch.
- **Recording schema**: pytest validates that a synthetic recording
  produces a JSONL where every line conforms to a JSON Schema, and the
  route manifest matches the actual sample count.
- **E-STOP behavior**: pytest spawns the safety supervisor against a
  fake `/cmd_vel` publisher; asserts that no command above zero passes
  through while latched.
- **Frontend mock-mode smoke**: vitest renders the dashboard with the
  synthetic generator and asserts every panel mounts without errors.

## 8. Rollout phases

1. **MVP** (this delivery): mock data wired end-to-end, ROS node
   skeletons running, all UI modules visible. No live training yet.
2. Phase 2: wire to live ROS 2 + Gazebo Sim; replace mock generator
   with real bridge.
3. Phase 3: Dreamer training sampler reads `data/replay_buffer/`,
   honors `priority=1.0` weighting.
4. Phase 4: route comparison metrics (Fréchet distance, success rate
   against reference).
