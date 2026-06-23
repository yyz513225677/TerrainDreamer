# Troubleshooting

Pairs of symptom → likely cause → fix. Add new entries chronologically.

| Symptom | Likely cause | Fix |
|---|---|---|
| `gdalinfo: command not found` | GDAL CLI not installed | `bash scripts/install_dependencies.sh` (installs `gdal-bin` + `python3-gdal`) |
| `ModuleNotFoundError: No module named 'rasterio'` | rasterio not installed | `sudo apt install python3-rasterio` or `pip install rasterio` |
| `prepare_dem_tile.sh` says "QGIS workflow required" | You did not pass `--center-x`/`--center-y` | Follow `docs/qgis_workflow.md` to pick coords, then re-run with them |
| `gdal_translate` warns `... values clipped` | Tile extent exceeds DEM extent | Re-pick centre away from the DEM edge or shrink `--size-meters` |
| `Gazebo Sim` fails: `Could not find heightmap PNG` | Heightmap not generated | Run `python3 scripts/normalize_heightmap.py …` before launching |
| `Gazebo Sim` fails: `Model 'lunar_dem_terrain' not found` | `GZ_SIM_RESOURCE_PATH` is missing the models dir | Source the package's `share` via colcon, or `export GZ_SIM_RESOURCE_PATH=$PWD/ros2_ws/src/lunar_south_pole_gazebo/models:$GZ_SIM_RESOURCE_PATH` |
| Heightmap appears flat in Gazebo | `vertical_scale_m` in the metadata YAML is zero | Re-run `normalize_heightmap.py` with `--clip-percentile 0,100` to avoid degenerate min==max |
| `/scan` topic absent | `ros_gz_bridge` not running, or topic name mismatch | Confirm `ros2 launch lunar_south_pole_gazebo bridge.launch.py`; inspect `config/bridge_jazzy.yaml` |
| `/cmd_vel` doesn't move the placeholder rover | diff-drive plugin not enabled in the model SDF | Verify `<plugin filename="gz-sim-diff-drive-system">` block in the rover SDF |
| Robot spawns inside terrain | Spawn height too low | Pass `--spawn-z` to `prepare_dem_tile.sh`, OR raise the `<pose>` Z in the spawn launch file |
| `gz sim` shows pitch-black scene | No directional light, or material is non-emissive | Increase `<ambient>` in the world, or check the directional light's elevation |
| `colcon build` fails: `package 'ros_gz_bridge' not found` | ROS not sourced | `source /opt/ros/jazzy/setup.bash` then `colcon build` |
| Launch warns "Jackal packages not found, using placeholder" | Expected on Jazzy currently | Either keep placeholder, or `apt install ros-jazzy-jackal-description` once Clearpath ships it |
| `dreamer_interface_node` only emits zero Twist | Stub behaviour — by design in stage 1 | Wire `src/terrain_dreamer/world_model/dreamer_policy.py` in stage 2 |
