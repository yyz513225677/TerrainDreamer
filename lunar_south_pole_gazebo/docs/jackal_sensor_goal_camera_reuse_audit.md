# Reuse Audit — Phase 2 (Jackal / Sensors / Goals / Chase camera / Route recording)

**Hard rule:** if a feature can be imported, installed, reused, configured,
launched, bridged, or wrapped from an existing official package, we do NOT
write it from scratch. This audit MUST be reviewed before any
implementation file is edited.

## Environment probe (2026-06-04)

| Capability | Status | Source / version |
|---|---|---|
| Official Clearpath Jackal packages (`jackal_description`, `jackal_control`) | **NOT installed for Jazzy** — apt has no `ros-jazzy-jackal-*`. Placeholder remains the working path. | n/a |
| Existing placeholder rover SDF | available, already has `gpu_lidar` + `imu` + diff-drive plugin | `models/jackal_placeholder/model.sdf` |
| `ros_gz_bridge` | installed (Jazzy 1.0.22) | `/opt/ros/jazzy/share/ros_gz_bridge` |
| `ros_gz_sim` (`create` node for spawning) | installed | `/opt/ros/jazzy/share/ros_gz_sim` |
| `gz-sim-diff-drive-system` (locomotion) | installed | `/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins/` |
| `gpu_lidar` / `imu` sensors | built into gz-sim 8.11 | `gz-sensors` |
| `camera` sensor type in gz | built in | `gz-sensors` |
| `visualization_msgs` (Marker / MarkerArray) | installed | `/opt/ros/jazzy/share/visualization_msgs` |
| `nav_msgs` (Odometry / Path) | installed | `/opt/ros/jazzy/share/nav_msgs` |
| `message_filters` (ApproximateTimeSynchronizer) | installed (Python via `rclpy`) | `/opt/ros/jazzy/lib/python3.12/site-packages/message_filters` |
| `tf2_ros` (`TransformListener`, `Buffer`) | installed | standard ROS 2 |
| `gz-sim-follow-actor-system` | available **but for animated actors only** — NOT a rigid-body chase camera | — |

## Per-feature reuse decisions

### F1 — Jackal spawning

* **Can it be reused?** Yes. `ros_gz_sim`'s `create` executable already
  spawns SDF/URDF into a running gz sim with `-file -name -x -y -z -Y …`.
  Phase 1 already wires this through `spawn_jackal.launch.py`.
* **Written here:** only the launch-file glue — argument plumbing,
  branch for `use_official_jackal`, safe-Z computation (read the
  `lunar_dem_terrain/model.sdf`'s `vscale`, add the configured
  `z_safe_above_dem`).
* **Why not custom:** we never build a custom spawn protocol; we call
  `ros_gz_sim create`.

### F2 — LiDAR / IMU sensors

* **Can it be reused?** Yes — stock gz sensors. The placeholder SDF
  already declares `<sensor name="lidar" type="gpu_lidar">` and
  `<sensor name="imu" type="imu">`.
* **Written here:** only the SDF parameter values (FOV 360°, range
  0.2 m–80 m, rate 10 Hz for lidar; 100 Hz for IMU) and the topic
  rename `/lunar_rover/*` → `/lunar_jackal/*`.
* **Why not custom:** never write a custom sensor simulator.

### F3 — Differential-drive control

* **Reuse:** `gz-sim-diff-drive-system` plugin. Already in the SDF.
* **Written here:** topic rename in the plugin block.

### F4 — ROS↔Gz bridging

* **Reuse:** `ros_gz_bridge parameter_bridge` (already used in Phase 1).
* **Written here:** add YAML entries for the new `/lunar_jackal/*` topic
  set and the chase-camera image/info topics. Keep the previous
  `/scan /imu /odom /cmd_vel` mapping as documented aliases.

### F5 — Destination markers

* **Can it be reused?** Yes — `visualization_msgs/MarkerArray`. RViz2
  and any dashboard can subscribe and render the markers natively.
* **Written here:** a thin publisher node that reads
  `config/lunar_goals.yaml` and emits a MarkerArray (cylinder beacon +
  TEXT_VIEW_FACING label per goal). No custom marker protocol.
* **Why not Gazebo-side models:** spawning per-goal SDF models would
  duplicate the visualization, complicate Gazebo loading, and provide
  no benefit beyond the RViz markers (which work in dashboards too).
* **Fallback:** if `visualization_msgs` is unavailable for any reason,
  the node degrades to a `std_msgs/String` JSON dump on
  `/lunar_jackal/goals/info`. We do not attempt SDF-side spawning in
  this stage (out of scope; tracked as a Phase 3 follow-up).

### F6 — Goal distance / arrival logic

* **Reuse:** stock `nav_msgs/Odometry`, `std_msgs/{Float32,String,Bool}`,
  and one Euclidean-distance line of NumPy.
* **Written here:** the FSM and dispatch in `goal_status_node.py`. The
  math is one line; the message types are stock.

### F7 — Chase camera (racing-game style)

* **Reuse path A (preferred):** add a `camera` sensor inside the
  robot SDF at a fixed offset behind/above `base_link`. This piggybacks
  on the same gz-sensors machinery as the lidar/IMU, and the camera
  pose stays rigidly attached to the robot — exactly the racing-game
  third-person behaviour the spec asks for. No interpolation code, no
  separate spawn, no custom transform listener.
* **Reuse path B (fallback):** Gazebo Sim's GUI **Follow target** menu
  item on the entity (`right-click on lunar_jackal → Follow`). This is
  always available and requires zero code, but it's not visible in
  ROS topics. Documented as an interactive option in the runbook.
* **Avoid:** writing a custom `tf2`-listening node that interpolates a
  free-floating chase camera entity. That is the brittle path.

### F8 — Route recording

* **Reuse:** `message_filters.ApproximateTimeSynchronizer` (ROS 2
  Python) is the standard sync primitive for stamped messages
  (`Odometry`, `Imu`). The latest non-stamped messages (`cmd_vel`,
  goal status) are cached and read at sync callbacks. Output format
  uses Python's stdlib `json` (JSONL) and PyYAML.
* **Written here:** the sync + serialise loop, ~150 LoC. No custom
  message types, no custom bag format.
* **Schema:** mirrors the spec literally — one JSON object per sample,
  the per-route YAML metadata file collects run-level fields.

### F9 — Route visualization

* **Reuse:** `nav_msgs/Path`. RViz already renders Path with no extra
  code.
* **Written here:** publisher node that converts incoming odometry +
  recorded JSONL into Path messages.

### F10 — Tests

* **Reuse:** pytest (already used in Phase 1).
* **Written here:** four test files exercising config presence,
  schema validity, topic-name consistency, and camera config presence.
  Tests do NOT spin rclpy or Gazebo.

## What this phase does NOT write from scratch

| Concern | Reused from existing packages |
|---|---|
| Robot URDF / spawn protocol | `ros_gz_sim create` |
| Sensor simulation | gz-sensors (`gpu_lidar`, `imu`, `camera`) |
| Locomotion plugin | `gz-sim-diff-drive-system` |
| Bridge | `ros_gz_bridge parameter_bridge` |
| Marker rendering | `visualization_msgs` (RViz / dashboards) |
| Path visualization | `nav_msgs/Path` |
| Topic synchronisation | `message_filters.ApproximateTimeSynchronizer` |
| Distance arithmetic | NumPy |
| File I/O | `json`, `PyYAML` |
| TF lookups | `tf2_ros` (not strictly required since we use odometry) |

## Summary

All Phase 2 features map to existing packages already installed on the
machine. The only new code is **configuration files**, **launch glue**,
**four small ROS 2 nodes** (goal marker, goal status, route recorder,
route visualizer), **SDF edits** (topic rename + chase-camera sensor),
**bridge YAML edits**, and **tests**. No physics, perception,
visualization-engine, or bridge code is reimplemented.
