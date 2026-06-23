"""Tests that the world / model / launch file inventory is complete
and that prepare_dem_tile.sh fails cleanly when the DEM is missing.
"""
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
PKG_DIR = HERE.parents[1]              # ros2_ws/src/lunar_south_pole_gazebo
PROJECT_ROOT = HERE.parents[4]         # lunar_south_pole_gazebo project root


EXPECTED_FILES = [
    "package.xml",
    "setup.py",
    "resource/lunar_south_pole_gazebo",
    "launch/lunar_south_pole.launch.py",
    "launch/spawn_jackal.launch.py",
    "launch/bridge.launch.py",
    "worlds/lunar_south_pole_empty.sdf",
    "worlds/lunar_south_pole_dem.sdf",
    "models/lunar_dem_terrain/model.config",
    "models/lunar_dem_terrain/model.sdf",
    "models/lunar_rocks/model.config",
    "models/lunar_rocks/model.sdf",
    "models/jackal_placeholder/model.config",
    "models/jackal_placeholder/model.sdf",
    "lunar_south_pole_gazebo/__init__.py",
    "lunar_south_pole_gazebo/dem_metadata_node.py",
    "lunar_south_pole_gazebo/terrain_manager_node.py",
    "lunar_south_pole_gazebo/hazard_status_node.py",
    "lunar_south_pole_gazebo/dreamer_interface_node.py",
]


def test_package_inventory_complete():
    missing = [r for r in EXPECTED_FILES if not (PKG_DIR / r).is_file()]
    assert not missing, f"missing files: {missing}"


def test_prepare_dem_script_fails_when_dem_absent(tmp_path):
    """`prepare_dem_tile.sh` must exit non-zero when input DEM is missing."""
    script = PROJECT_ROOT / "scripts" / "prepare_dem_tile.sh"
    assert script.is_file()
    fake_in = tmp_path / "does_not_exist.tif"
    fake_out = tmp_path / "out.tif"
    res = subprocess.run(
        ["bash", str(script),
         "--input", str(fake_in),
         "--output", str(fake_out),
         "--center-x", "0", "--center-y", "0"],
        capture_output=True, text=True, check=False,
        env={**os.environ, "PATH": os.environ.get("PATH", "")},
    )
    assert res.returncode != 0, \
        "expected non-zero exit when DEM is missing; got 0"


def test_inspect_dem_script_fails_when_dem_absent(tmp_path):
    script = PROJECT_ROOT / "scripts" / "inspect_dem.sh"
    fake = tmp_path / "nope.tif"
    res = subprocess.run(["bash", str(script), str(fake)],
                         capture_output=True, text=True, check=False)
    assert res.returncode != 0
