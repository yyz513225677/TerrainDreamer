"""Tests for the route recorder's sample schema.

The recorder's `make_sample` is a pure function; we verify the output
matches the spec's literal example shape and degrades gracefully when
optional fields (cmd_vel, goal, imu, collision) are missing.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
PKG_DIR = HERE.parent.parent
sys.path.insert(0, str(PKG_DIR))
from lunar_south_pole_gazebo.route_recorder_node import (  # noqa: E402
    make_sample, _quat_to_yaw)


def _minimal_sample(**overrides):
    base = dict(
        timestamp=0.0, route_id="rt", robot_name="lunar_jackal",
        pose={"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        odom={"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0},
        imu={}, cmd_vel={}, goal={}, collision=False)
    base.update(overrides)
    return make_sample(**base)


def test_top_level_keys():
    s = _minimal_sample()
    assert set(s.keys()) == {
        "timestamp", "route_id", "robot_name", "pose", "odom",
        "imu", "cmd_vel", "goal", "collision",
    }


def test_pose_shape():
    s = _minimal_sample(pose={"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 0.4})
    assert s["pose"] == {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 0.4}


def test_imu_defaults():
    s = _minimal_sample()
    assert s["imu"]["orientation"] == [0.0, 0.0, 0.0, 1.0]
    assert s["imu"]["angular_velocity"] == [0.0, 0.0, 0.0]
    assert s["imu"]["linear_acceleration"] == [0.0, 0.0, 0.0]


def test_cmd_vel_defaults():
    s = _minimal_sample()
    assert s["cmd_vel"] == {"linear_x": 0.0, "angular_z": 0.0}


def test_goal_defaults():
    s = _minimal_sample()
    assert s["goal"]["goal_id"] == ""
    assert s["goal"]["distance_to_goal"] == 0.0
    assert s["goal"]["goal_reached"] is False


def test_collision_is_bool():
    s = _minimal_sample(collision=True)
    assert s["collision"] is True


def test_handles_missing_imu_gracefully():
    """Recording must NOT depend on IMU — fields default cleanly when IMU
    delivers nothing."""
    s = _minimal_sample(imu={})
    assert len(s["imu"]["orientation"]) == 4
    assert all(v == 0.0 for v in s["imu"]["angular_velocity"])


def test_quat_to_yaw_identity():
    # Identity quaternion → yaw = 0
    assert abs(_quat_to_yaw(0.0, 0.0, 0.0, 1.0)) < 1e-9


def test_quat_to_yaw_90deg():
    # 90° around Z → quaternion (0, 0, sin45, cos45)
    import math
    s2 = math.sin(math.pi / 4)
    yaw = _quat_to_yaw(0.0, 0.0, s2, s2)
    assert abs(yaw - math.pi / 2) < 1e-6
