# Clearpath J100 Jackal — Add to an Already-Running Gazebo Sim World

This adds an **official Clearpath J100** (generated from
`~/clearpath/robot.yaml`) to a Gazebo Sim world that is ALREADY RUNNING.
No new map geometry is created or modified.

## Files added

| File | Purpose |
|---|---|
| `~/clearpath/robot.yaml` | Clearpath config — J100 + Velodyne VLP-32C + IMU |
| `ros2_ws/src/lunar_south_pole_gazebo/launch/spawn_clearpath_jackal.launch.py` | Thin wrapper around `clearpath_gz robot_spawn.launch.py` |

## Required packages (Jazzy)

```bash
sudo apt install -y --no-install-recommends \
  ros-${ROS_DISTRO:-jazzy}-clearpath-simulator \
  ros-${ROS_DISTRO:-jazzy}-clearpath-gz \
  ros-${ROS_DISTRO:-jazzy}-clearpath-desktop \
  ros-${ROS_DISTRO:-jazzy}-clearpath-config \
  ros-${ROS_DISTRO:-jazzy}-clearpath-generator-gz \
  ros-${ROS_DISTRO:-jazzy}-velodyne-description
```

To switch distros, set `ROS_DISTRO=humble` before sourcing.

## Important caveats (not all spec items map 1:1 to the official stack)

| Spec item | Reality | Outcome |
|---|---|---|
| Velodyne VLP-32C driver / calibration | `velodyne_pointcloud` ships VLP-32C calibration `VeloView-VLP-32C.yaml` and `model: 32C` | ✅ pipeline is true 32C |
| VLP-32C **visual** URDF | Clearpath `velodyne_lidar.urdf.xacro` hard-includes `VLP-16.urdf.xacro` — no 32C visual macro | ⚠ visual mesh = VLP-16 puck, but topic/scan = VLP-32C |
| VectorNav VN-100 IMU | NOT a Clearpath-supported model. The schema only accepts: `microstrain_imu`, `microstrain_gq7`, `redshift_um7`, `chrobotics_um6`, `phidgets_spatial` | ⚠ substituted with `microstrain_imu`; edit `sensors.imu[0].model` in `robot.yaml` to pick a different one |

## 1. Find the running world's name

```bash
gz topic -l | grep '/world/' | head
# e.g. /world/lunar_south_pole/clock  →  world name is "lunar_south_pole"
```

## 2. Spawn the robot

```bash
source /opt/ros/jazzy/setup.bash
source ~/Documents/Leo/terrain_dreamer/lunar_south_pole_gazebo/ros2_ws/install/setup.bash

ros2 launch lunar_south_pole_gazebo spawn_clearpath_jackal.launch.py \
  setup_path:=$HOME/clearpath/ \
  world:=lunar_south_pole \
  x:=0.0 y:=0.0 z:=1.0 yaw:=0.0 \
  rviz:=false
```

The launch only calls `clearpath_gz robot_spawn.launch.py`; it does NOT
start gz sim. It also starts the platform ros_gz_bridge so the
generated topics surface on the ROS 2 side.

## 3. Verify topics

```bash
# Platform namespace defined in robot.yaml: j100_0001
NS=/j100_0001

ros2 topic info ${NS}/cmd_vel                            # velocity command (in)
ros2 topic info ${NS}/platform/odom                      # odometry
ros2 topic info ${NS}/sensors/lidar3d_0/points           # VLP-32C point cloud
ros2 topic info ${NS}/sensors/imu_0/data                 # IMU
ros2 topic list | grep -E "/tf"                          # transforms
```

## 4. Drive the rover

```bash
ros2 topic pub /j100_0001/cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.3, y: 0.0, z: 0.0},
    angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 10
```

## 5. Visualise

```bash
# Quick image / pointcloud view
ros2 run rqt_image_view rqt_image_view
ros2 run rviz2 rviz2

# Or relaunch with rviz:=true (it spawns RViz alongside)
ros2 launch lunar_south_pole_gazebo spawn_clearpath_jackal.launch.py rviz:=true
```

## 6. Build

```bash
cd ~/Documents/Leo/terrain_dreamer/lunar_south_pole_gazebo/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select lunar_south_pole_gazebo --symlink-install
source install/setup.bash
```

## 7. Full demo (existing world + Clearpath spawn)

```bash
# Terminal A — start your existing world (no Clearpath robot yet)
ros2 launch lunar_south_pole_gazebo lunar_south_pole.launch.py \
  use_dem:=true use_jackal:=false      # suppress placeholder jackal

# Terminal B — spawn the official Clearpath J100
source /opt/ros/jazzy/setup.bash
source ~/Documents/Leo/terrain_dreamer/lunar_south_pole_gazebo/ros2_ws/install/setup.bash
ros2 launch lunar_south_pole_gazebo spawn_clearpath_jackal.launch.py \
  setup_path:=$HOME/clearpath/ world:=lunar_south_pole z:=1.0
```
