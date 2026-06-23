"""Tests for static configuration files.

Verifies:
  * config/*.yaml exist and parse;
  * dem_config.yaml exposes the documented keys;
  * the metadata loader in dem_metadata_node raises cleanly when the
    target YAML is missing.
"""
import sys
from pathlib import Path

import pytest
import yaml

# Repo root: this test file is at
#   ros2_ws/src/lunar_south_pole_gazebo/test/test_dem_config.py
HERE = Path(__file__).resolve()
PKG_DIR = HERE.parent.parent           # …/lunar_south_pole_gazebo (the package)
PROJECT_ROOT = PKG_DIR.parents[2]      # …/lunar_south_pole_gazebo (project)

sys.path.insert(0, str(PKG_DIR))
from lunar_south_pole_gazebo.dem_metadata_node import load_metadata  # noqa: E402


def test_config_files_exist():
    for name in ("dem_config.yaml", "bridge_humble.yaml",
                 "bridge_jazzy.yaml", "terrain_params.yaml"):
        p = PROJECT_ROOT / "config" / name
        assert p.is_file(), f"missing config file: {p}"


def test_dem_config_has_required_keys():
    p = PROJECT_ROOT / "config" / "dem_config.yaml"
    data = yaml.safe_load(p.read_text())
    for k in ("raw_dem_path", "processed_dem_path",
              "heightmap_png_path", "metadata_yaml_path",
              "crop", "normalize"):
        assert k in data, f"dem_config.yaml missing key: {k}"
    assert data["crop"]["center_x"] is None, \
        "We must NOT bake Shackleton coordinates into the config."
    assert data["crop"]["center_y"] is None


def test_terrain_params_has_rocks_section():
    p = PROJECT_ROOT / "config" / "terrain_params.yaml"
    data = yaml.safe_load(p.read_text())
    assert "rocks" in data
    assert "count" in data["rocks"]


def test_load_metadata_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_metadata(missing)


def test_load_metadata_missing_keys_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"tile_width_m": 1.0}))
    with pytest.raises(ValueError):
        load_metadata(bad)


def test_load_metadata_happy_path(tmp_path):
    good = tmp_path / "ok.yaml"
    good.write_text(yaml.safe_dump({
        "tile_width_m": 2048.0,
        "tile_height_m": 2048.0,
        "samples_x": 1025,
        "samples_y": 1025,
        "min_elevation_m": -1.0,
        "max_elevation_m": 1.0,
        "vertical_scale_m": 2.0,
    }))
    data = load_metadata(good)
    assert data["samples_x"] == 1025
