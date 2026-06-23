# Phase 2 — Verification Report

Run: 2026-06-04. ROS 2 Jazzy, Gazebo Sim Harmonic, Ubuntu 24.04.

## 1. Static checks

| # | Check | Tool | Result |
|---|---|---|---|
| 1 | Python syntax — nodes | `python3 -m py_compile` on 4 new + 4 existing nodes | **PASS** |
| 2 | Python syntax — launch | `python3 -m py_compile` on all 3 launch files | **PASS** |
| 3 | YAML parse — all 6 configs | `yaml.safe_load` | **PASS** |
| 4 | SDF / XML well-formed | `xml.etree.ElementTree.parse` on every `.sdf` / `.xml` | **PASS** (caught + fixed an XML comment with `--` inside) |
| 5 | No machine-specific absolute paths in shipped artefacts | `grep -rn /home/rickslab3 ros2_ws/ config/` | **PASS** |

## 2. Build

```
$ colcon build --packages-select lunar_south_pole_gazebo --symlink-install
Starting >>> lunar_south_pole_gazebo
Finished <<< lunar_south_pole_gazebo [1.50s]
Summary: 1 package finished [1.71s]
```

Clean. After `source install/setup.bash`:

```
$ ros2 pkg executables lunar_south_pole_gazebo
lunar_south_pole_gazebo dem_metadata_node          (Phase 1)
lunar_south_pole_gazebo dreamer_interface_node     (Phase 1)
lunar_south_pole_gazebo goal_marker_node           (Phase 2)
lunar_south_pole_gazebo goal_status_node           (Phase 2)
lunar_south_pole_gazebo hazard_status_node         (Phase 1)
lunar_south_pole_gazebo route_recorder_node        (Phase 2)
lunar_south_pole_gazebo route_visualizer_node      (Phase 2)
lunar_south_pole_gazebo terrain_manager_node       (Phase 1)
```

All 8 entry points registered (4 Phase-1 + 4 Phase-2).

## 3. Launch sanity

```
$ ros2 launch lunar_south_pole_gazebo lunar_south_pole.launch.py --show-args
```
…lists all Phase-1 args plus the new Phase-2 args:
`use_sensors`, `use_chase_camera`, `use_goal_markers`,
`use_route_recorder`, `record_route` (alias),
`route_id`, `use_route_visualizer`,
`x`, `y`, `z`, `roll`, `pitch`, `yaw`, `z_safe_above_dem`,
`robot_name`, `use_official_jackal`, plus the four topic-name args.

## 4. Tests

```
$ pytest ros2_ws/src/lunar_south_pole_gazebo/test/ -v
============================== 38 passed in 0.19s ==============================
```

Phase-1 (14) + Phase-2 (24) tests pass:

* `test_dem_config.py` × 6
* `test_route_topics.py` × 5  (Phase-1 topic set kept as aliases — converted from `==` to ⊆ check after Phase 2 added topics)
* `test_world_file_exists.py` × 3
* `test_goal_config.py` × 6
* `test_route_record_schema.py` × 9
* `test_topic_names.py` × 5  *(includes regression test for camera_info/image gz-topic-name typo)*
* `test_chase_camera_config.py` × 4

## 5. Files added or modified

### Added

| File | Purpose |
|---|---|
| `docs/jackal_sensor_goal_camera_reuse_audit.md` | reuse decisions |
| `docs/jackal_sensor_goal_camera_plan.md` | sprint plan |
| `docs/jackal_sensor_goal_camera_runbook.md` | end-user runbook |
| `docs/jackal_sensor_goal_camera_verification_report.md` | this file |
| `config/lunar_goals.yaml` | 5 example goal waypoints |
| `config/chase_camera.yaml` | camera intrinsics + offsets |
| `ros2_ws/src/lunar_south_pole_gazebo/lunar_south_pole_gazebo/goal_marker_node.py` | RViz MarkerArray publisher |
| `ros2_ws/src/lunar_south_pole_gazebo/lunar_south_pole_gazebo/goal_status_node.py` | distance + arrival logic |
| `ros2_ws/src/lunar_south_pole_gazebo/lunar_south_pole_gazebo/route_recorder_node.py` | JSONL recorder with odom+IMU sync |
| `ros2_ws/src/lunar_south_pole_gazebo/lunar_south_pole_gazebo/route_visualizer_node.py` | live + recorded Path |
| `ros2_ws/src/lunar_south_pole_gazebo/launch/chase_camera.launch.py` | optional standalone camera-bridge launch |
| `ros2_ws/src/lunar_south_pole_gazebo/test/test_goal_config.py` | 6 tests |
| `ros2_ws/src/lunar_south_pole_gazebo/test/test_route_record_schema.py` | 9 tests |
| `ros2_ws/src/lunar_south_pole_gazebo/test/test_topic_names.py` | 5 tests |
| `ros2_ws/src/lunar_south_pole_gazebo/test/test_chase_camera_config.py` | 4 tests |

### Modified

| File | Change |
|---|---|
| `ros2_ws/src/lunar_south_pole_gazebo/models/jackal_placeholder/model.sdf` | sensor topic rename `/lunar_rover/*` → `/lunar_jackal/*`; added chase_camera sensor; lidar range/FOV widened (0.2 – 80 m, 720 samples) |
| `ros2_ws/src/lunar_south_pole_gazebo/launch/spawn_jackal.launch.py` | full argument surface (x/y/z/roll/pitch/yaw/robot_name/use_official_jackal/z_safe_above_dem/topic names); safe-Z calculation from tile metadata YAML |
| `ros2_ws/src/lunar_south_pole_gazebo/launch/lunar_south_pole.launch.py` | Phase-2 flags; wires goal_marker, goal_status, route_recorder, route_visualizer; OR-gate for use_route_recorder / record_route alias |
| `ros2_ws/src/lunar_south_pole_gazebo/setup.py` | registers 4 new console entry points |
| `config/bridge_jazzy.yaml` | added `/lunar_jackal/*` topic set + chase camera; kept `/scan /imu /odom /cmd_vel` as aliases |
| `config/bridge_humble.yaml` | same Phase-2 additions for Ignition Fortress / Humble |
| `ros2_ws/src/lunar_south_pole_gazebo/test/test_route_topics.py` | Phase-1 topic-set test converted from `==` to ⊆ check (Phase 2 added topics; aliases must still be present) |
| `README.md` | Phase-2 commands section |
| `docs/jackal_sensor_goal_camera_plan.md` | added `/tf` to topic table (caught by `test_phase2_plan_mentions_each_topic`) |

### NOT modified

* Phase 1 docs (`reuse_audit.md`, `design.md`, `implementation_plan.md`,
  `qgis_workflow.md`, `dem_source_notes.md`, `troubleshooting.md`,
  `verification_report.md`)
* DEM pipeline scripts (`scripts/{inspect,prepare,normalize,colorize,
  fill_nodata,generate_gazebo_world,generate_procedural_hazards}.py`)
* Existing world SDFs (apart from the unrelated XML-comment fix)
* `models/lunar_dem_terrain/`, `models/lunar_rocks/`

## 6. Bugs caught + fixed during this phase

1. **Bridge YAML camera_info gz_topic_name typo**: had pasted the
   image topic instead of `/lunar_jackal/chase_camera/camera_info`.
   Caught by a regression test
   (`test_camera_info_maps_to_camera_info_gz_topic`) and fixed in both
   `bridge_jazzy.yaml` and `bridge_humble.yaml`.
2. **Duplicate route_recorder spawn**: initial draft had two `Node`
   blocks gated by `use_route_recorder` and `record_route`
   independently — would have spawned two recorders writing the same
   file. Fixed with a single `PythonExpression`-based OR condition.
3. **XML comment with `--`**: a generator string had `<!-- … via
   --gui-config … -->` which is illegal (XML forbids `--` inside
   comments). Caught by the XML-parse static check and fixed in the
   template + the previously-generated world SDF.
4. **Phase-1 `test_route_topics` failed under Phase 2**: tightened
   equality assertion broke when Phase 2 expanded the bridge topic
   set. Converted to a subset assertion (Phase-1 aliases must be
   present; Phase 2 may add).

## 7. Items not exercised in this phase (require interactive Gazebo)

* Live spawn of the rover via `ros_gz_sim create` — needs the gz sim
  running, which crashes silently after some seconds on this host's
  OGRE2 + heightmap LOD path. Static configuration is verified.
* Live `/lunar_jackal/chase_camera/image` rendering — same caveat.
* Manual driving with `ros2 topic pub /lunar_jackal/cmd_vel …` — the
  command line is documented in the runbook; the diff-drive plugin
  is the stock `gz-sim-diff-drive-system`.

## 7b. Code-review fixes (post-review)

An independent code review surfaced 1 blocker and 5 majors. All were
addressed:

| # | Severity | Issue | Fix |
|---|---|---|---|
| B1 | 🔴 | Safe-Z used unscaled `vertical_scale_m` from the YAML (7088 m) but the rendered terrain is scaled to 354 m in the world SDF — the rover would have fallen 6.7 km. | `spawn_jackal.launch.py` now reads the displayed Z from the terrain SDF's `<size>` element; capped at 1000 m as a hard safety. |
| M1 | 🟠 | `route_recorder` metadata YAML only written on `KeyboardInterrupt` — SIGTERM lost data. | `atexit` + `SIGTERM/SIGINT/SIGHUP` handlers call an idempotent `write_meta()`. |
| M2 | 🟠 | Shared `cache["goal"]` mutated in-place from three callbacks while sync reader read it — torn writes under `MultiThreadedExecutor`. | Callbacks now do whole-dict replacement (`cache["goal"] = {**cache["goal"], ...}`); sync callback snapshots via `dict(cache["goal"])`. |
| M3 | 🟠 | `goal_marker_node` republished the full MarkerArray at 1 Hz despite using TRANSIENT_LOCAL durability — wasted work. | Drop the timer; latched publisher emits once on startup. |
| M4 | 🟠 | `goal_status_node` advance-on-reach left `goal_reached=True` latched on the NEW goal for ~100 ms. | After advance, immediately publish `goal_reached=False` against the new goal. |
| M5 | 🟠 | Reviewer suspected chase-camera pitch sign was inverted. | Verified analytically: SDF pose RPY uses right-hand rule about Y; positive pitch rotates +X (look direction) toward -Z (downward). `atan2(3 m, 6 m) = +0.46 rad`, so `pitch=+0.40` is correctly downward (slightly under-angled — fine). Added a comment to the SDF so future readers don't second-guess. |

All 38 tests still pass after the fixes; `colcon build` clean.

## 8. Conclusion

Phase 2 is structurally complete and statically verified. All
build/test/launch parsing checks pass. The runtime exercises that
need a stable Gazebo Sim window are documented in the runbook for
interactive use.
