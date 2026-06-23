"""Tests for the chase camera configuration."""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[4]
CFG = PROJECT_ROOT / "config" / "chase_camera.yaml"
ROVER_SDF = (PROJECT_ROOT / "ros2_ws" / "src" / "lunar_south_pole_gazebo"
             / "models" / "jackal_placeholder" / "model.sdf")


def test_chase_camera_yaml_exists():
    assert CFG.is_file()


def test_chase_camera_yaml_required_keys():
    data = yaml.safe_load(CFG.read_text())
    for k in ("offset", "orientation", "intrinsics", "update_rate_hz",
              "ros_image_topic", "ros_camera_info_topic"):
        assert k in data, f"chase_camera.yaml missing {k}"
    for k in ("x", "y", "z"):
        assert k in data["offset"]
    for k in ("horizontal_fov_rad", "width_px", "height_px",
              "near_clip", "far_clip"):
        assert k in data["intrinsics"]


def test_chase_camera_spec_values():
    """The spec mandates -6 m behind / 3 m above, ~80° FOV."""
    data = yaml.safe_load(CFG.read_text())
    assert data["offset"]["x"] == -6.0
    assert data["offset"]["z"] == 3.0
    assert 1.22 <= data["intrinsics"]["horizontal_fov_rad"] <= 1.57


def test_rover_sdf_has_chase_camera():
    """The chase camera must be a sensor inside the rover SDF — the
    rigid-attach option from the reuse audit, not a free-floating model."""
    sdf = ROVER_SDF.read_text()
    assert 'name="chase_camera"' in sdf, \
        "rover SDF missing chase_camera sensor"
    assert '/lunar_jackal/chase_camera/image' in sdf, \
        "rover SDF should publish chase camera on /lunar_jackal/chase_camera/image"
