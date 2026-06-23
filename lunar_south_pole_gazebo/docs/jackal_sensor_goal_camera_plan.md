# Implementation Plan — Phase 2

Pre-flight: this plan is read AFTER
[`docs/jackal_sensor_goal_camera_reuse_audit.md`](jackal_sensor_goal_camera_reuse_audit.md).
Both files are required before any implementation file is edited.

Phase 1 (DEM pipeline + colorised heightmap + rectangular fill) is
DONE. **This phase does not touch the DEM/heightmap/world generation
pipeline.**

## 1. Order of operations

| Step | Output | Rule |
|---|---|---|
| 1 | docs/jackal_sensor_goal_camera_reuse_audit.md (✓ done) | reuse decisions |
| 2 | This file | sprint plan |
| 3 | config/lunar_goals.yaml + chase_camera.yaml | declarative configs (no code yet) |
| 4 | SDF edits — placeholder rover topic rename + chase camera sensor | reuse stock sensors |
| 5 | spawn_jackal.launch.py — full arg surface + safe-Z + namespace | reuse `ros_gz_sim create` |
| 6 | goal_marker_node.py + goal_status_node.py | publish MarkerArray, distance, reached |
| 7 | route_recorder_node.py + route_visualizer_node.py | sync via message_filters; emit Path |
| 8 | bridge_*.yaml + lunar_south_pole.launch.py — wire it all together | reuse `ros_gz_bridge` |
| 9 | setup.py — register 4 new entry points | reuse Phase 1 ament_python layout |
| 10 | 4 pytest files + verification report | reuse Phase 1 test style |
| 11 | Runbook + README updates | end-user docs |
| 12 | Code review | independent agent pass |

## 2. Topic & namespace contract

Everything robot-side goes under `/lunar_jackal/*`. This means:

| ROS topic | Gz topic | Direction | Type |
|---|---|---|---|
| `/lunar_jackal/scan` | `/lunar_jackal/scan` | Gz→ROS | `sensor_msgs/LaserScan` |
| `/lunar_jackal/imu` | `/lunar_jackal/imu` | Gz→ROS | `sensor_msgs/Imu` |
| `/lunar_jackal/odom` | `/lunar_jackal/odom` | Gz→ROS | `nav_msgs/Odometry` |
| `/lunar_jackal/cmd_vel` | `/lunar_jackal/cmd_vel` | ROS→Gz | `geometry_msgs/Twist` |
| `/lunar_jackal/chase_camera/image` | `/lunar_jackal/chase_camera/image` | Gz→ROS | `sensor_msgs/Image` |
| `/lunar_jackal/chase_camera/camera_info` | `/lunar_jackal/chase_camera/camera_info` | Gz→ROS | `sensor_msgs/CameraInfo` |
| `/clock` | `/clock` | Gz→ROS | `rosgraph_msgs/Clock` |
| `/tf` | `/tf` | Gz→ROS | `tf2_msgs/TFMessage` |

Plus the previous topics `/scan /imu /odom /cmd_vel` are **kept** in
the bridge YAML as documented aliases (each maps from the same gz
topic), so any Phase-1 consumer keeps working.

## 3. ROS-side topics (Phase 2 nodes)

| Topic | Type | Producer | Purpose |
|---|---|---|---|
| `/lunar_jackal/goals` | `visualization_msgs/MarkerArray` | goal_marker_node | RViz beacons + labels |
| `/lunar_jackal/current_goal` | `std_msgs/String` | goal_status_node | active goal_id |
| `/lunar_jackal/goal_distance` | `std_msgs/Float32` | goal_status_node | metres |
| `/lunar_jackal/goal_reached` | `std_msgs/Bool` | goal_status_node | within radius |
| `/lunar_jackal/live_path` | `nav_msgs/Path` | route_visualizer_node | live odometry path |
| `/lunar_jackal/recorded_path` | `nav_msgs/Path` | route_visualizer_node | from saved JSONL |
| `/collision_status` | `std_msgs/Bool` | (optional, future) | best-effort contact |

## 4. Files (matched to spec sections)

| Spec § | File | LoC budget | Type |
|---|---|---|---|
| § A | `launch/spawn_jackal.launch.py` (modify) | ~150 | launch |
| § B | `models/jackal_placeholder/model.sdf` (modify — rename topics, add chase cam) | +30 | SDF |
| § C | `config/lunar_goals.yaml` (new) | ~50 | YAML |
| § C | `lunar_south_pole_gazebo/goal_marker_node.py` (new) | ~120 | node |
| § D | `lunar_south_pole_gazebo/goal_status_node.py` (new) | ~110 | node |
| § E | `config/chase_camera.yaml` (new) | ~30 | YAML |
| § E | `launch/chase_camera.launch.py` (new, optional thin wrapper) | ~50 | launch |
| § F | `lunar_south_pole_gazebo/route_recorder_node.py` (new) | ~180 | node |
| § G | `lunar_south_pole_gazebo/route_visualizer_node.py` (new) | ~120 | node |
| § H | `launch/lunar_south_pole.launch.py` (modify — add Phase-2 flags) | +60 | launch |
| § I | `config/bridge_jazzy.yaml` (modify), `bridge_humble.yaml` (modify) | +50 each | YAML |
| § H | `setup.py` (modify — 4 new entry points) | +8 | setup |
| § L | `test/test_goal_config.py` (new) | ~70 | test |
| § L | `test/test_route_record_schema.py` (new) | ~80 | test |
| § L | `test/test_topic_names.py` (new) | ~70 | test |
| § L | `test/test_chase_camera_config.py` (new) | ~40 | test |
| § K | `docs/jackal_sensor_goal_camera_runbook.md` (new) | ~180 | docs |
| Verif. | `docs/jackal_sensor_goal_camera_verification_report.md` (new) | ~80 | docs |

## 5. Synchronisation strategy (route recorder)

The two high-rate stamped messages are **odom (20 Hz) + IMU (100 Hz)**.
They get an `ApproximateTimeSynchronizer` with `slop=0.05` and queue=20.
On every sync callback we write one JSONL line and combine in the
latest cached values of:

* `cmd_vel` (latest `geometry_msgs/Twist`, no header → cached)
* `current_goal` / `goal_distance` / `goal_reached` (3 std_msgs latches)
* `collision_status` (best-effort, default False if not seen)

This is the standard ROS pattern; we do NOT invent a new sync algorithm.

## 6. Safe-Z spawn

The DEM in this stage is displayed at 10 km × 10 km × 354 m vscale
(see Phase 1 §7). At centre (0,0) the heightmap pixel value lands the
terrain somewhere between 0 and 354 m. To avoid spawn-clipping:

* Read `data/metadata/south_pole_tile.yaml`'s `vertical_scale_m`.
* Add a fixed `z_safe_above_dem` (default 2.0 m).
* Compute `z = vertical_scale_m + z_safe_above_dem` and pass it to
  `ros_gz_sim create -z`. The rover falls onto the heightmap (lunar
  gravity 1.62 m/s², so the drop is gentle).

## 7. Definition of Done

* All files in §4 exist.
* `python3 -m py_compile` passes on every new `.py`.
* All YAML parses with `yaml.safe_load`.
* All SDF parses with `xml.etree.ElementTree.parse`.
* New tests pass (`pytest test/`).
* `colcon build --packages-select lunar_south_pole_gazebo` succeeds.
* `ros2 pkg executables lunar_south_pole_gazebo` lists all four new
  nodes plus the Phase-1 four.
* Bridge YAML mentions every topic in §2.
* Verification report is written and lists each check.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Chase-camera sensor pushes gz-sim memory over the heightmap-LOD cliff | Camera image size kept modest (640×360 @ 15 Hz); fallback documented as Gazebo GUI Follow |
| Topic rename breaks Phase-1 consumers | Keep `/scan /imu /odom /cmd_vel` as bridge aliases |
| Spawn-Z below the DEM crashes the rover into terrain | Safe-Z derived from metadata YAML, configurable |
| ApproximateTimeSynchronizer drops samples when rates diverge | Set `slop=0.05`, `queue_size=20`; log a warning every N skipped pairs |
| Goal markers invisible in RViz if no fixed frame set | Document required RViz fixed frame (`odom`) in the runbook |
