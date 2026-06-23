# lunar_south_pole_gazebo

Stage 1 backend simulator: reproducible **ROS 2 + Gazebo Sim** world
generated from a real **NASA PGDA LOLA Lunar South Pole DEM**
(`LDEM_80S_20MPP_ADJ.TIF`, 20 m/px, 80°S – 90°S, polar stereographic
metres).

There is no frontend in this stage. The output is a Gazebo world
plus the standard ROS 2 sensor/control topics, ready for any
downstream policy (Dreamer included).

---

## Import-first policy

Before adding code to this project, check whether the feature is
already provided by:

* **GDAL** (`gdalinfo`, `gdal_translate`, `gdalwarp`)
* **rasterio** + **NumPy** + **Pillow**
* **Gazebo Sim** heightmap + sensor + diff-drive plugins
* **ros_gz_bridge** (topic mapping via YAML)
* **sensor_msgs / nav_msgs / geometry_msgs / std_msgs / tf2_msgs**
* the official Clearpath Jackal packages (if installed)

If the answer is yes, use the existing thing — do **not**
re-implement. The full reuse rationale is in
[`docs/reuse_audit.md`](docs/reuse_audit.md).

---

## Project layout

```
lunar_south_pole_gazebo/
├── docs/         design + reuse audit + workflows + troubleshooting
├── scripts/      thin wrappers around GDAL, Pillow, SDF generation
├── config/       YAML config (DEM defaults, bridge mappings, hazards)
├── ros2_ws/src/  one ament_python package: lunar_south_pole_gazebo
├── data/         raw_dem / processed_dem / heightmaps / metadata
└── outputs/      screenshots + logs from Gazebo sessions
```

The design contract is [`docs/design.md`](docs/design.md). The
file-by-file plan is
[`docs/implementation_plan.md`](docs/implementation_plan.md).

---

## Run-from-scratch sequence

### 1. Install dependencies

```bash
bash scripts/install_dependencies.sh
```

Installs `gdal-bin python3-gdal python3-numpy python3-yaml
python3-pil python3-rasterio` plus `ros-${ROS_DISTRO:-jazzy}-ros-gz-bridge`.

### 2. Place the raw DEM

```bash
# Manually copy/symlink the file you obtained from NASA PGDA:
cp /path/to/LDEM_80S_20MPP_ADJ.TIF data/raw_dem/
ls -lh data/raw_dem/LDEM_80S_20MPP_ADJ.TIF
```

This project does **not** download NASA data — see
[`docs/dem_source_notes.md`](docs/dem_source_notes.md).

### 3. Inspect the DEM

```bash
bash scripts/inspect_dem.sh data/raw_dem/LDEM_80S_20MPP_ADJ.TIF
cat data/metadata/dem_info.txt | head
```

### 4. Crop a tile

If you already know your region's centre in projected metres:

```bash
bash scripts/prepare_dem_tile.sh \
  --input  data/raw_dem/LDEM_80S_20MPP_ADJ.TIF \
  --output data/processed_dem/shackleton_tile.tif \
  --center-x  <PS_metres_x> \
  --center-y  <PS_metres_y> \
  --size-meters 2048 \
  --samples 1025
```

If you don't, follow [`docs/qgis_workflow.md`](docs/qgis_workflow.md).

### 5. Generate the heightmap PNG + metadata

```bash
python3 scripts/normalize_heightmap.py \
  --input  data/processed_dem/shackleton_tile.tif \
  --output data/heightmaps/shackleton_heightmap.png \
  --meta   data/metadata/shackleton_tile.yaml
```

### 6. Generate the Gazebo world

```bash
python3 scripts/generate_gazebo_world.py \
  --meta   data/metadata/shackleton_tile.yaml \
  --pkg    ros2_ws/src/lunar_south_pole_gazebo

# Optional: scatter procedural rocks (LOLA is too coarse for rover-scale rocks)
python3 scripts/generate_procedural_hazards.py \
  --config config/terrain_params.yaml \
  --pkg    ros2_ws/src/lunar_south_pole_gazebo
```

### 7. Build the ROS 2 package

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select lunar_south_pole_gazebo
source install/setup.bash
```

### 8. Launch

```bash
# Full sim: world + rover + bridge
ros2 launch lunar_south_pole_gazebo lunar_south_pole.launch.py \
  use_dem:=true use_jackal:=true use_bridge:=true

# Just spawn (assumes a world is already running)
ros2 launch lunar_south_pole_gazebo spawn_jackal.launch.py x:=0.0 y:=0.0 yaw:=0.0

# Just bridge (assumes Gazebo is running)
ros2 launch lunar_south_pole_gazebo bridge.launch.py bridge_config:=auto
```

### 9. Sanity check `/cmd_vel`

```bash
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/Twist \
  "{linear: {x: 0.3}, angular: {z: 0.0}}"
```

The placeholder rover should drive forward at 0.3 m/s.

---

## Topics (stage 1)

| Direction | Topic | Type |
|---|---|---|
| Gz→ROS | `/clock` | `rosgraph_msgs/Clock` |
| Gz→ROS | `/scan` | `sensor_msgs/LaserScan` |
| Gz→ROS | `/imu` | `sensor_msgs/Imu` |
| Gz→ROS | `/odom` | `nav_msgs/Odometry` |
| Gz→ROS | `/tf` | `tf2_msgs/TFMessage` |
| Gz→ROS | `/collision_status` | `std_msgs/Bool` (best-effort) |
| ROS→Gz | `/cmd_vel` | `geometry_msgs/Twist` |
| ROS→ROS | `/dashboard/reset` | `std_msgs/Empty` |
| ROS→ROS | `/dashboard/estop` | `std_msgs/Bool` |

Aliases documented for downstream consumers (not duplicated by the
bridge — use `<remap>` if needed):
`/lunar_rover/scan`, `/lunar_rover/imu`, `/lunar_rover/odom`,
`/lunar_rover/cmd_vel`.

---

## Using the official Clearpath Jackal (when available)

`spawn_jackal.launch.py` checks `ament_index_python` for a
`jackal_description` share at runtime. If found, it spawns the
official URDF/xacro; otherwise it spawns
`models/jackal_placeholder/`. To migrate:

```bash
sudo apt install ros-${ROS_DISTRO}-jackal-description \
                 ros-${ROS_DISTRO}-jackal-control
# then re-launch — no project edits required.
```

---

## Connecting Dreamer later (stage 2)

`dreamer_interface_node.py` subscribes to `/scan`, `/imu`, `/odom`
and publishes `/cmd_vel`. The control loop currently emits zero
Twist with `# TODO: call dreamer_policy.act(obs)` markers. To wire
in the parent repo's Dreamer:

```python
# in dreamer_interface_node.py
from terrain_dreamer.world_model.dreamer_policy import DreamerPolicy
self._policy = DreamerPolicy.load(checkpoint_path)
twist = self._policy.act(self._latest_obs)
```

The bridge, sensor topology, and ROS plumbing do not change.

---

## Tests

```bash
cd ros2_ws/src/lunar_south_pole_gazebo
python3 -m pytest test/ -v
```

Tests do **not** spin Gazebo or rclpy — they verify config presence,
topic-name consistency, and clean failure when prerequisites are
missing.

---

## Phase 2 — Jackal, sensors, goals, chase camera, route recording

See the [Phase-2 runbook](docs/jackal_sensor_goal_camera_runbook.md)
for the full walkthrough. Quick reference:

```bash
# Full Phase-2 system
ros2 launch lunar_south_pole_gazebo lunar_south_pole.launch.py \
  use_dem:=true use_jackal:=true use_sensors:=true \
  use_goal_markers:=true use_chase_camera:=true \
  use_route_recorder:=true

# Drive the rover
ros2 topic pub /lunar_jackal/cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.25, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.15}}" -r 10

# Verify sensors
ros2 topic echo /lunar_jackal/scan       # LiDAR
ros2 topic echo /lunar_jackal/imu        # IMU
ros2 topic echo /lunar_jackal/odom       # Odometry

# Goal status
ros2 topic echo /lunar_jackal/current_goal
ros2 topic echo /lunar_jackal/goal_distance
ros2 topic echo /lunar_jackal/goal_reached

# Recorded routes land here:
ls data/routes/
```

Phase-2 topics (full list in
[`docs/jackal_sensor_goal_camera_plan.md`](docs/jackal_sensor_goal_camera_plan.md) §2 + §3):

| Topic | Type |
|---|---|
| `/lunar_jackal/scan` | `sensor_msgs/LaserScan` |
| `/lunar_jackal/imu` | `sensor_msgs/Imu` |
| `/lunar_jackal/odom` | `nav_msgs/Odometry` |
| `/lunar_jackal/cmd_vel` | `geometry_msgs/Twist` |
| `/lunar_jackal/chase_camera/image` | `sensor_msgs/Image` |
| `/lunar_jackal/chase_camera/camera_info` | `sensor_msgs/CameraInfo` |
| `/lunar_jackal/goals` | `visualization_msgs/MarkerArray` |
| `/lunar_jackal/current_goal` | `std_msgs/String` |
| `/lunar_jackal/goal_distance` | `std_msgs/Float32` |
| `/lunar_jackal/goal_reached` | `std_msgs/Bool` |
| `/lunar_jackal/live_path` | `nav_msgs/Path` |
| `/lunar_jackal/recorded_path` | `nav_msgs/Path` |

Phase-1 aliases (`/scan`, `/imu`, `/odom`, `/cmd_vel`) are kept in the
bridge YAML for backwards compatibility.

## Troubleshooting

See [`docs/troubleshooting.md`](docs/troubleshooting.md).
