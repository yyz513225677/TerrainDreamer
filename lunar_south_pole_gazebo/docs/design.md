# Design — `lunar_south_pole_gazebo`

## 1. Purpose

Stage 1 backend simulator: build a reproducible ROS 2 + Gazebo Sim
world from a real NASA PGDA LOLA Lunar South Pole DEM
(`LDEM_80S_20MPP_ADJ.TIF`, 20 m/px, 80°S–90°S, polar stereographic
metres). Provide standard ROS 2 sensor + control topics so a rover
policy (Dreamer, MPC, …) can later be wired in without touching the
sim.

This stage explicitly does **not** include a frontend, a trainer, or
any custom physics or rendering.

## 2. Data flow

```
NASA PGDA LOLA DEM   (user places at data/raw_dem/LDEM_80S_20MPP_ADJ.TIF)
        |
        |  scripts/inspect_dem.sh         -> data/metadata/dem_info.txt
        v
   gdalinfo summary
        |
        |  scripts/prepare_dem_tile.sh    (GDAL: gdal_translate -projwin)
        v
   data/processed_dem/shackleton_tile.tif        (1025×1025, 2048 m tile)
        |
        |  scripts/normalize_heightmap.py (rasterio + NumPy + Pillow)
        v
   data/heightmaps/shackleton_heightmap.png   (16-bit grey)
   data/metadata/shackleton_tile.yaml         (extents, vertical_scale_m, …)
        |
        |  scripts/generate_gazebo_world.py (SDF string from YAML)
        v
   models/lunar_dem_terrain/{model.config, model.sdf}
   worlds/lunar_south_pole_dem.sdf
        |
        |  ros2 launch lunar_south_pole_gazebo lunar_south_pole.launch.py
        v
   Gazebo Sim (Harmonic) running the DEM world
        |
        |  ros_gz_bridge  (config/bridge_jazzy.yaml)
        v
   ROS 2 topics: /scan /imu /odom /tf /clock /cmd_vel ...
```

Each arrow above is one already-existing tool. The only new artefacts
on this graph are *configuration files* (the YAML metadata, the SDF
strings, the bridge YAML, the launch files).

## 3. Coordinate frames

* The DEM is in **south polar stereographic metres**. We do **not**
  reproject — we keep the projected metres as the Gazebo world XY.
* After cropping to a 2048 m tile centred on a user-chosen (x, y) in
  PS-metres, the heightmap PNG covers exactly `[xmin..xmax] × [ymin..ymax]`
  in the world frame. The SDF `<size>` tag is set to
  `(tile_width_m, tile_height_m, vertical_scale_m)`.
* Robot origin (0, 0, 0) is placed at the *centre* of the heightmap by
  default (configurable via the launch arg `x:=`, `y:=`).
* `tf` chain: `world → odom → base_link → {laser, imu_link, …}` with
  `odom → base_link` published by the Gazebo diff-drive plugin.

## 4. Topic catalogue

| Direction | Topic | Type | Source |
|---|---|---|---|
| Gz→ROS | `/clock` | `rosgraph_msgs/Clock` | gz-sim |
| Gz→ROS | `/scan` | `sensor_msgs/LaserScan` | gpu_lidar sensor |
| Gz→ROS | `/imu` | `sensor_msgs/Imu` | imu sensor |
| Gz→ROS | `/odom` | `nav_msgs/Odometry` | diff-drive plugin |
| Gz→ROS | `/tf` | `tf2_msgs/TFMessage` | diff-drive + sensor tfs |
| Gz→ROS | `/collision_status` (best effort) | `std_msgs/Bool` | optional contact sensor |
| ROS→Gz | `/cmd_vel` | `geometry_msgs/Twist` | downstream policy |
| ROS→ROS | `/dashboard/reset` | `std_msgs/Empty` | downstream |
| ROS→ROS | `/dashboard/estop` | `std_msgs/Bool` | downstream |

Documented aliases (consumer-side remap if a downstream package
expects them): `/lunar_rover/{scan,imu,odom,cmd_vel}`.

The bridge YAML lists exactly these. Aliases are documented in the
README but the bridge does not duplicate the topics; downstream code
that needs the alias should remap.

## 5. SDF layout

```
worlds/lunar_south_pole_dem.sdf
  <world name="lunar_south_pole">
    <gravity>0 0 -1.62</gravity>          <!-- moon -->
    <atmosphere><type>adiabatic</type></atmosphere>  <!-- no air -->
    <scene><ambient>0.02 0.02 0.02 1</ambient>
           <background>0 0 0 1</background>           <!-- black sky -->
           ... </scene>
    <light type="directional"> ... low-angle solar ...</light>
    <include><uri>model://lunar_dem_terrain</uri></include>
    <include><uri>model://lunar_rocks</uri></include>  <!-- if generated -->
    <plugin filename="gz-sim-physics-system" ... />
    <plugin filename="gz-sim-sensors-system" ... />
    <plugin filename="gz-sim-scene-broadcaster-system" ... />
  </world>
```

```
models/lunar_dem_terrain/model.sdf
  <model name="lunar_dem_terrain"><static>true</static><link name="link">
    <collision name="col"><geometry><heightmap>
      <uri>../heightmaps/shackleton_heightmap.png</uri>
      <size>2048 2048 vertical_scale_m</size>
      <sampling>2</sampling>
    </heightmap></geometry></collision>
    <visual name="vis"><geometry><heightmap>...</heightmap></geometry>
      <material><ambient>0.18 0.18 0.18 1</ambient></material>
    </visual>
  </link></model>
```

Notes:
* The PNG URI is *relative* so the model is portable.
* `<size>` Z component is the `(max - min)` elevation from the YAML —
  this is the per-pixel-value-1 vertical scale.
* `<static>true</static>` — the terrain doesn't move.

## 6. ROS 2 package

`ros2_ws/src/lunar_south_pole_gazebo/` is an `ament_python` package
that ships:

* four utility nodes (metadata, terrain manager, hazard status,
  Dreamer interface) — these are **glue**: they read the metadata
  YAML, publish status, and provide hooks. None of them implement
  physics, perception, or planning.
* three launch files (full sim, spawn-only, bridge-only).
* the worlds + models directories so Gazebo can find them via
  `GZ_SIM_RESOURCE_PATH`.
* a `test/` directory with pytest tests for config presence,
  topic-name contract, and missing-file behaviour.

## 7. Failure modes (and how each is handled)

| Failure | Detection | Response |
|---|---|---|
| User hasn't placed the raw DEM | `scripts/inspect_dem.sh` `[ ! -f "$1" ]` | exit 2 with message pointing to `data/raw_dem/README.md` |
| GDAL not installed | `command -v gdalinfo` | exit 1 with `install_dependencies.sh` hint |
| User omitted `--center-x/y` | argparse | print QGIS workflow path, exit 0 (instructional) |
| `shackleton_tile.tif` not present when normalising | `Path.exists()` | clean error |
| Heightmap PNG not yet generated when launching | launch precondition | log and continue with `lunar_south_pole_empty.sdf` |
| `ros_gz_bridge` not installed | bridge.launch.py | log warning, skip bridge node |
| `ROS_DISTRO` unset | launch arg `bridge_config:=auto` | fall back to `bridge_jazzy.yaml` (default) |
| Jackal packages absent | `spawn_jackal.launch.py` | spawn `jackal_placeholder` SDF, log a warning |

## 8. Test scope

Tests do **not** spin Gazebo or rclpy. They verify:

* every file listed in `docs/implementation_plan.md §6` exists;
* `dem_metadata_node` raises a clear error when the YAML is missing;
* the topic-name table is consistent between `bridge_jazzy.yaml`,
  the README, and the design doc;
* `prepare_dem_tile.sh` exits non-zero when DEM is missing.

## 9. Out of scope (stage 1)

* Frontend (explicitly excluded).
* Training Dreamer.
* Real-Jackal hardware bring-up.
* High-fidelity dust/illumination shading.
* Multi-rover.
