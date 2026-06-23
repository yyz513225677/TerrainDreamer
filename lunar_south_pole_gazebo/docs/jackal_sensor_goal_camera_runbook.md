# Runbook — Phase 2: Jackal, sensors, goals, chase camera, route recording

This runbook assumes Phase 1 is done (DEM tile + heightmap + world SDF
+ colcon build succeed) and `ros-jazzy-ros-gz-bridge` /
`ros-jazzy-ros-gz-sim` are installed.

## 1. Source the workspace

```bash
cd lunar_south_pole_gazebo/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

If you have an active project venv, deactivate it first: this project
uses system Python so `rclpy` resolves correctly.

## 2. Launch the full system

```bash
ros2 launch lunar_south_pole_gazebo lunar_south_pole.launch.py \
  use_dem:=true \
  use_jackal:=true \
  use_sensors:=true \
  use_goal_markers:=true \
  use_chase_camera:=true \
  use_route_recorder:=true
```

Effect:
1. The DEM world (`lunar_south_pole_dem.sdf`) loads in Gazebo Sim.
2. `spawn_jackal.launch.py` spawns the placeholder rover at the safe-Z
   computed from `data/metadata/south_pole_tile.yaml`.
3. `ros_gz_bridge parameter_bridge` starts with `config/bridge_jazzy.yaml`,
   wiring every `/lunar_jackal/*` topic plus the Phase-1 aliases.
4. The four utility nodes (Phase 1) plus four new feature nodes
   (Phase 2 — goal marker, goal status, route recorder, route
   visualizer) come up.

## 3. Verify Jackal spawn

```bash
ros2 topic list | grep lunar_jackal
# expect: /lunar_jackal/scan /lunar_jackal/imu /lunar_jackal/odom
#         /lunar_jackal/cmd_vel /lunar_jackal/chase_camera/image
#         /lunar_jackal/chase_camera/camera_info
#         /lunar_jackal/current_goal /lunar_jackal/goal_distance
#         /lunar_jackal/goal_reached /lunar_jackal/goals
#         /lunar_jackal/live_path /lunar_jackal/recorded_path
ros2 topic hz /lunar_jackal/odom                # ~20 Hz
```

## 4. Verify each sensor

LiDAR:
```bash
ros2 topic echo --once /lunar_jackal/scan | head -30
ros2 topic hz /lunar_jackal/scan                # ~10 Hz
```

IMU:
```bash
ros2 topic echo --once /lunar_jackal/imu | head -20
ros2 topic hz /lunar_jackal/imu                 # ~100 Hz
```

Odometry:
```bash
ros2 topic echo --once /lunar_jackal/odom | head -20
```

## 5. Goals on the map

The goal_marker_node publishes
`/lunar_jackal/goals` (`visualization_msgs/MarkerArray`).
In RViz: add a MarkerArray display, topic = `/lunar_jackal/goals`,
fixed frame = `odom`. You'll see the five beacons + arrival
rings + labels.

```bash
ros2 topic echo /lunar_jackal/current_goal
ros2 topic echo /lunar_jackal/goal_distance
ros2 topic echo /lunar_jackal/goal_reached
```

To change the active goal at runtime:
```bash
ros2 topic pub --once /lunar_jackal/set_goal std_msgs/msg/String \
  "data: goal_03_safe_basin"
```

Goals are loaded from `config/lunar_goals.yaml` — edit the file,
rebuild the package, and re-launch to change them. There are no
hard-coded coordinates in any node.

## 6. Chase camera (third-person racing-game view)

The chase camera is a `<sensor type="camera">` rigidly attached to
the rover SDF at `(-6, 0, 3)` with `pitch ≈ 23°`. It tracks the
robot's pose at the simulation tick rate — no interpolation, no tf2
listener.

* Image: `/lunar_jackal/chase_camera/image` (640×360 @ 15 Hz)
* CameraInfo: `/lunar_jackal/chase_camera/camera_info`

View the stream:
```bash
ros2 run rqt_image_view rqt_image_view /lunar_jackal/chase_camera/image
```

Alternative (no camera sensor needed): in the running Gazebo Sim
window, right-click on the rover entity → **Follow target**. This
binds the GUI's main camera to the rover and is the fallback path
recorded in the reuse audit. The follow-target view is interactive
only — no ROS topic is produced.

## 7. Drive the rover

```bash
ros2 topic pub /lunar_jackal/cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.25, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.15}}" \
  -r 10
```

(`/cmd_vel` also works thanks to the bridge alias.)

The rover will drive forward at 0.25 m/s while turning left at 0.15 rad/s.

## 8. Route recording

Recording starts automatically when launched with
`use_route_recorder:=true` (or the alias `record_route:=true`). Output:

* `data/routes/route_YYYYMMDD_HHMMSS.jsonl` — one JSON object per
  synchronised odom+IMU sample (so the route timeline aligns with IMU
  observations even when their rates differ).
* `data/routes/route_YYYYMMDD_HHMMSS.yaml` — per-run metadata
  (start/end timestamp, sample count, source topics).

`Ctrl-C` (or `ros2 lifecycle shutdown` of the recorder) flushes the
metadata file. The JSONL is line-buffered, so a crash mid-run still
leaves a valid file.

A specific `route_id` can be requested:
```bash
ros2 launch lunar_south_pole_gazebo lunar_south_pole.launch.py \
  use_route_recorder:=true route_id:=teleop_north_loop_001
```

## 9. Why "route is recorded from odom, not from the IMU"

* The IMU provides orientation, angular velocity, and linear
  acceleration — strapdown signals.
* Position over time comes from integrating the IMU **OR** reading it
  out of odometry. Odometry integration is already done by
  `gz-sim-diff-drive-system` and published on `/lunar_jackal/odom`
  with much less drift than naive IMU integration would have.
* The route recorder consumes both — `ApproximateTimeSynchronizer`
  pairs the latest odometry pose with the latest IMU sample (slop
  ≤ 50 ms), so each JSONL line has both the "where am I" (from odom)
  and the "what is my body feeling" (from IMU).

## 10. Live + recorded path visualization

The route_visualizer_node publishes:

* `/lunar_jackal/live_path` (`nav_msgs/Path`) — built from incoming
  `/lunar_jackal/odom`.
* `/lunar_jackal/recorded_path` (`nav_msgs/Path`) — only published if
  the node is launched with the `recorded_jsonl` parameter pointing
  at a saved JSONL file.

In RViz: add a Path display for each, set the colour, fixed frame
`odom`.

```bash
# Show a previously recorded route alongside the live drive
ros2 run lunar_south_pole_gazebo route_visualizer_node --ros-args \
  -p recorded_jsonl:=$(pwd)/data/routes/route_20260604_120000.jsonl
```

## 11. Stopping the recorder explicitly

If you launched without `use_route_recorder:=true` and want to add it
later (without restarting everything):

```bash
ros2 run lunar_south_pole_gazebo route_recorder_node --ros-args \
  -p route_id:=manual_session
```

`Ctrl-C` writes the YAML metadata and closes the JSONL cleanly.

## 12. Using route data for Dreamer training

The saved JSONL is a flat per-step record. Each sample has the same
keys (timestamp, route_id, pose, odom, imu, cmd_vel, goal, collision)
so a parent-repo training script can read it directly:

```python
import json
with open("data/routes/route_<id>.jsonl") as fh:
    samples = [json.loads(l) for l in fh if l.strip()]
# samples[i]['pose']['yaw']  etc.
```

The `route_id` and `robot_name` are repeated per sample for safe
shuffling/concatenation; the YAML metadata file groups them at the
run level.

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| No topics seen | source `install/setup.bash`; confirm `ros2 daemon stop && ros2 daemon start` |
| Rover spawns inside the heightmap | leave `z` at `auto` so safe-Z is read from the metadata YAML |
| Markers invisible in RViz | set Fixed Frame to `odom` (not `map`) |
| `route_recorder_node` writes 0 samples | confirm `/lunar_jackal/odom` AND `/lunar_jackal/imu` are publishing |
| `goal_reached` always false | check `arrival_radius_m` in `config/lunar_goals.yaml` (default 3 m) |
| Want to see chase camera but no image | confirm bridge entries for `/lunar_jackal/chase_camera/{image,camera_info}` are running |

## 14. Inputs/outputs at a glance

| Node | Subscribes | Publishes |
|---|---|---|
| goal_marker_node | (none) | `/lunar_jackal/goals` (latched MarkerArray) |
| goal_status_node | `/lunar_jackal/odom`, `/lunar_jackal/set_goal` | `/lunar_jackal/current_goal`, `/lunar_jackal/goal_distance`, `/lunar_jackal/goal_reached` |
| route_recorder_node | `/lunar_jackal/odom` + `/lunar_jackal/imu` (sync), `/lunar_jackal/cmd_vel`, goal topics, `/collision_status` | JSONL + YAML in `data/routes/` |
| route_visualizer_node | `/lunar_jackal/odom` (optional `recorded_jsonl` param) | `/lunar_jackal/live_path`, `/lunar_jackal/recorded_path` |
