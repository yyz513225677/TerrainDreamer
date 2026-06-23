# Reuse Audit — `lunar_south_pole_gazebo`

**Hard rule:** If a feature can be imported, installed, reused, bridged, or
configured, we do not write it from scratch. This audit must be reviewed
before any implementation file is edited.

## Environment probe results (2026-06-04)

| Tool | Status | Version |
|---|---|---|
| `gdalinfo` / `gdal_translate` / `gdalwarp` | **missing** | — install via `gdal-bin` |
| `python3-gdal` (osgeo) | **missing** | — install via `python3-gdal` |
| `rasterio` | **missing** | — install via apt `python3-rasterio` or pip |
| Pillow | available | 12.2.0 |
| NumPy | available | 1.26.4 |
| PyYAML | available | 6.0.3 |
| Gazebo Sim (gz sim) | available | 8.11.0 (Harmonic) |
| ROS 2 | available | Jazzy (/opt/ros/jazzy) |
| `ros_gz_bridge` | available | Jazzy build |
| Clearpath Jackal packages | **missing** | placeholder required |

## Per-feature reuse decisions

### F1 — DEM metadata inspection

* **Can it be imported?** Yes.
* **Which package?** `gdal-bin` (`gdalinfo` CLI).
* **How used?** `scripts/inspect_dem.sh` is a one-line wrapper that
  redirects `gdalinfo -stats -hist` output into `data/metadata/dem_info.txt`.
* **Why custom code?** None — pure wrapper.
* **Fallback if missing?** The script exits with `gdalinfo: command not
  found — run scripts/install_dependencies.sh`.

### F2 — DEM cropping (projected metres)

* **Can it be imported?** Yes.
* **Which package?** `gdal-bin` (`gdal_translate -projwin`, optionally
  `gdalwarp -te` if reprojection is later needed).
* **How used?** `scripts/prepare_dem_tile.sh` calls `gdal_translate`
  with `-projwin xmin ymax xmax ymin -outsize SAMPLES SAMPLES`.
* **Why custom code?** None.
* **Fallback?** If `--center-x/--center-y` are not provided, the script
  prints the QGIS manual workflow from `docs/qgis_workflow.md`. We do
  **not** invent Shackleton coordinates.

### F3 — GeoTIFF reading (Python)

* **Can it be imported?** Yes.
* **Which package?** `rasterio` (preferred) **or** `osgeo.gdal`.
* **How used?** `scripts/normalize_heightmap.py` opens the cropped tile,
  reads band 1 into a NumPy array, queries georeferencing/transform.
* **Why custom code?** None — we only call `rasterio.open(...)`.
* **Fallback?** If `rasterio` import fails, the script transparently
  falls back to `from osgeo import gdal` with the same call shape.

### F4 — Heightmap normalisation (numerics)

* **Can it be imported?** Partially. NumPy provides everything.
* **Which package?** NumPy.
* **How used?** `(arr - min) / (max - min) * 65535` and `arr.astype(np.uint16)`.
* **Why custom code?** This is a 3-line transform — no library provides
  "elevations → Gazebo-ready 0..65535 unsigned heightmap" as a single
  call. The arithmetic itself is from NumPy.
* **Fallback?** N/A — NumPy is already installed.

### F5 — PNG export

* **Can it be imported?** Yes.
* **Which package?** Pillow (`PIL.Image`).
* **How used?** `Image.fromarray(arr_u16, mode="I;16").save(out_png)`.
* **Why custom code?** None.
* **Fallback?** If Pillow is unavailable (it is, however, present),
  `imageio.imwrite(out_png, arr_u16)` is used.

### F6 — Gazebo terrain rendering

* **Can it be imported?** Yes — Gazebo Sim has a built-in heightmap
  geometry (`<geometry><heightmap>...</heightmap></geometry>`).
* **Which package?** `gz-sim` (Harmonic) ships `heightmap.sdf` as an
  example world; the engine uses Ogre2 + Bullet to render and collide
  16-bit grey heightmaps.
* **How used?** `scripts/generate_gazebo_world.py` writes an SDF that
  references our PNG and the metadata YAML for size/scale.
* **Why custom code?** Only the SDF *string* (a config), not the
  rendering engine.
* **Fallback?** None needed — Gazebo Harmonic is installed.

### F7 — Procedural rock hazards

* **Can it be imported?** Partially. Gazebo provides primitives
  (`<box>`, `<sphere>`); placing them is up to us.
* **Which package?** Gazebo Sim SDF primitives.
* **How used?** `scripts/generate_procedural_hazards.py` writes SDF
  `<include>` snippets for N rocks at random poses inside the
  rover-spawn safe radius.
* **Why custom code?** Placement policy is project-specific. We do not
  write a "rock engine" — each rock is just `<model><link><visual>
  <geometry><sphere|box>...</sphere|box></geometry></visual></link></model>`.
* **Fallback?** If `config/terrain_params.yaml` is missing, the script
  uses hard-coded sensible defaults (50 rocks, radii 0.05–0.4 m).

### F8 — ROS↔Gazebo bridging

* **Can it be imported?** Yes — installed.
* **Which package?** `ros_gz_bridge` (Jazzy build).
* **How used?** A YAML config file enumerates topic mappings; the
  `parameter_bridge` node consumes it. `bridge.launch.py` selects
  `bridge_humble.yaml` vs `bridge_jazzy.yaml` based on `$ROS_DISTRO`.
* **Why custom code?** Only the YAML config.
* **Fallback?** None — Jazzy is installed and `ros_gz_bridge` is built.

### F9 — Jackal robot model + drivers

* **Can it be imported?** Currently **no** — `apt list --installed 2>/dev/null
  | grep -i jackal` returns nothing.
* **Which package?** Would be `ros-jazzy-jackal-description`,
  `ros-jazzy-jackal-control` if/when published for Jazzy. Until then we
  ship a *clearly marked* placeholder under
  `models/jackal_placeholder/`.
* **How used?** `spawn_jackal.launch.py` first tries to spawn the
  official URDF, then falls back to the placeholder SDF.
* **Why custom code?** Only the placeholder model + the fallback
  glue. README and the model.config say "PLACEHOLDER — replace with
  official Clearpath Jackal when Jazzy packages land."
* **Fallback?** The placeholder is the fallback.

### F10 — LiDAR / IMU sensors

* **Can it be imported?** Yes — Gazebo built-in sensors.
* **Which package?** `gz-sensors` (loaded by `gz-sim`). Sensor types
  `gpu_lidar`, `imu`. Topics flow through `ros_gz_bridge` as
  `sensor_msgs/LaserScan` and `sensor_msgs/Imu`.
* **How used?** Sensor tags inside the rover model SDF.
* **Why custom code?** None — only XML wiring.
* **Fallback?** None.

### F11 — Differential drive / skid steer

* **Can it be imported?** Yes — `libgz-sim8-diff-drive-system.so` plugin
  ships with Gazebo Harmonic.
* **Which package?** `gz-sim`.
* **How used?** Included in the placeholder rover SDF as a `<plugin>`
  tag. Subscribes to `/cmd_vel` via bridge.
* **Why custom code?** Only the `<plugin>` config block.
* **Fallback?** Skid-steer plugin (`libgz-sim8-tracked-vehicle-system.so`)
  if differential drive is unsuitable for the Jackal-like footprint.

### F12 — Dreamer policy

* **Can it be imported?** Out of scope for this stage. Existing project
  has `src/terrain_dreamer/` — we expose a placeholder ROS 2 node that
  subscribes to sensor topics and publishes zero `Twist` on
  `/cmd_vel`. Connecting the real Dreamer is the next stage.
* **Which package?** Future: `terrain_dreamer.world_model.dreamer_policy`
  from the parent repo.
* **How used?** `dreamer_interface_node.py` has clear `# TODO:` blocks
  marking the interface points.
* **Why custom code?** Only the stub node skeleton.
* **Fallback?** The stub IS the fallback for stage 1.

### F13 — ROS 2 messages

* **Can it be imported?** Yes — already present in Jazzy.
* **Which packages?** `sensor_msgs`, `nav_msgs`, `geometry_msgs`,
  `std_msgs`, `tf2_msgs`. No custom `.msg` files in this stage.
* **How used?** Topic types in launch + nodes use the stock messages.
* **Why custom code?** None.
* **Fallback?** N/A.

## Summary

| Category | Reused from existing packages | Written here (config/glue only) |
|---|---|---|
| Raster I/O | GDAL, rasterio, NumPy, Pillow | Wrapper scripts |
| Simulation | Gazebo Sim heightmap, gz-sim diff-drive, gz-sensors lidar/IMU | SDF strings |
| ROS bridging | ros_gz_bridge | Bridge YAML, launch files |
| Messages | sensor_msgs, nav_msgs, geometry_msgs, std_msgs | None |
| Robot | (intended: official Jackal) | Placeholder SDF only because Jackal not packaged for Jazzy yet |
| Policy | (future: terrain_dreamer Dreamer) | Stub node with TODOs |

**Conclusion:** the only files we author from scratch are configuration
files, launch files, four wrapper nodes, glue scripts that call
existing tools, the placeholder rover, package metadata, docs, and
tests. No raster engine, no bridge, no sensor sim, no physics engine.
