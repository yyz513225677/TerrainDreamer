# Implementation Plan — `lunar_south_pole_gazebo`

## 0. Order of operations

The reuse audit (`docs/reuse_audit.md`) and design (`docs/design.md`)
are written **before** any implementation file. This plan
operationalises them.

## 1. Phases

### Phase A — Scaffolding (no business logic)
* Create the directory tree per spec.
* Write the three documentation files first
  (`reuse_audit.md`, `design.md`, this file).
* Write `README.md` skeleton with the import-first policy at the top.

### Phase B — Configuration files (declarative, no code yet)
* `config/dem_config.yaml` — default DEM path + tile defaults.
* `config/bridge_humble.yaml` and `config/bridge_jazzy.yaml` — the
  ros_gz_bridge mapping table.
* `config/terrain_params.yaml` — procedural-hazard defaults.

### Phase C — Wrapper scripts
* `scripts/install_dependencies.sh` — `apt-get install` + clear
  instructions; uses `${ROS_DISTRO:-jazzy}`.
* `scripts/inspect_dem.sh` — calls `gdalinfo`, redirects to
  `data/metadata/dem_info.txt`.
* `scripts/prepare_dem_tile.sh` — calls `gdal_translate -projwin`;
  prints QGIS instructions when coords absent.
* `scripts/normalize_heightmap.py` — `rasterio → NumPy → Pillow`.
* `scripts/generate_gazebo_world.py` — YAML → SDF string.
* `scripts/generate_procedural_hazards.py` — config → SDF includes.

### Phase D — ROS 2 package
* `package.xml`, `setup.py`, resource marker.
* Four nodes (see design §6) — each ≤120 LoC, no physics.
* Three launch files with the launch-argument set spec'd in the
  requirements.
* SDF worlds + model directories.

### Phase E — Tests
* Three pytest files covering config presence, topic-name consistency,
  missing-file handling.

### Phase F — Verification
* `python3 -m py_compile` on every `.py` in the project.
* `bash -n` on every `.sh`.
* `xmllint --noout` on every `.sdf` / `.xacro` if `xmllint` is present.
* Confirm `package.xml`, `setup.py`, launch files, world files,
  README's command list are all internally consistent.
* Write `docs/verification_report.md` summarising the above.

### Phase G — Code review
* Self-review against:
  - Import-first hard rule (any place we rolled our own?)
  - Hard-coded absolute paths (must be 0)
  - Tests run without rclpy / without Gazebo
  - SDF + bridge YAML topic names match design §4

## 2. File-by-file checklist

| File | Phase | Purpose | LoC budget |
|---|---|---|---|
| `docs/reuse_audit.md` | A | Reuse decisions | ✓ |
| `docs/design.md` | A | Architecture | ✓ |
| `docs/implementation_plan.md` | A | This file | ✓ |
| `docs/qgis_workflow.md` | A | Manual crop instructions | ~80 |
| `docs/dem_source_notes.md` | A | NASA PGDA notes (no URLs) | ~40 |
| `docs/troubleshooting.md` | A | Failure→fix table | ~80 |
| `README.md` | A | Top-level entry point | ~180 |
| `config/dem_config.yaml` | B | Defaults | ~30 |
| `config/bridge_humble.yaml` | B | ROS-Humble mapping | ~50 |
| `config/bridge_jazzy.yaml` | B | ROS-Jazzy mapping | ~50 |
| `config/terrain_params.yaml` | B | Hazard defaults | ~30 |
| `scripts/install_dependencies.sh` | C | apt + warnings | ~80 |
| `scripts/inspect_dem.sh` | C | gdalinfo wrapper | ~30 |
| `scripts/prepare_dem_tile.sh` | C | gdal_translate wrapper | ~120 |
| `scripts/normalize_heightmap.py` | C | rasterio→PNG | ~150 |
| `scripts/generate_gazebo_world.py` | C | YAML→SDF | ~180 |
| `scripts/generate_procedural_hazards.py` | C | rocks SDF | ~120 |
| `ros2_ws/src/lunar_south_pole_gazebo/package.xml` | D | ament metadata | ~30 |
| `ros2_ws/src/lunar_south_pole_gazebo/setup.py` | D | ament_python entry points | ~50 |
| `…/lunar_south_pole_gazebo/__init__.py` | D | empty | 0 |
| `…/dem_metadata_node.py` | D | publishes YAML metadata once | ~80 |
| `…/terrain_manager_node.py` | D | terrain re/loading hooks | ~80 |
| `…/hazard_status_node.py` | D | aggregates hazard topics | ~80 |
| `…/dreamer_interface_node.py` | D | stub I/O contract | ~80 |
| `…/launch/lunar_south_pole.launch.py` | D | full sim | ~120 |
| `…/launch/spawn_jackal.launch.py` | D | rover spawn | ~80 |
| `…/launch/bridge.launch.py` | D | bridge-only | ~80 |
| `…/worlds/lunar_south_pole_empty.sdf` | D | fallback world | ~80 |
| `…/worlds/lunar_south_pole_dem.sdf` | D | DEM world (generated, but a stub also kept) | ~80 |
| `…/models/lunar_dem_terrain/model.{config,sdf}` | D | generated stub | ~60 |
| `…/models/lunar_rocks/model.{config,sdf}` | D | generated stub | ~30 |
| `…/models/jackal_placeholder/model.{config,sdf}` | D | placeholder rover | ~120 |
| `…/test/test_*.py` | E | three test files | ~150 total |

## 3. Definition of Done

* All files in §2 exist.
* `python3 -m py_compile` passes on every .py file.
* `bash -n` passes on every .sh.
* No file contains the literal `/home/rickslab3` outside docs.
* The topic-name set in `bridge_jazzy.yaml` matches design §4.
* `docs/verification_report.md` is written and lists each check + pass/fail.

## 4. Risks + mitigations

| Risk | Mitigation |
|---|---|
| GDAL not installed at run time | install_dependencies.sh + clear error in every wrapper |
| Heightmap PNG path is brittle across machines | use `model://…` URIs + `GZ_SIM_RESOURCE_PATH` env in launch files |
| `ros_gz_bridge` topic-type spelling drifts between ROS releases | ship distinct `bridge_humble.yaml` and `bridge_jazzy.yaml` |
| Jackal packages eventually arrive on Jazzy and break the placeholder fallback | `spawn_jackal.launch.py` prefers the official URDF first, placeholder second |
| Latest LOLA DEM might re-tile (`LDEM_80S_…` is one of several) | YAML config-driven, no path hard-coded inside Python |
