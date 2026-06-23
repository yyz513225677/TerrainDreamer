"""Lightweight schema validation for one RouteSample dict.

Keeps an external jsonschema dep optional — we do a structural walk by
hand so the recorder can run with only stdlib + numpy. Returns a list of
error strings; empty list ⇒ valid.
"""

from __future__ import annotations

from typing import Any


_NUMBER = (int, float)

REQUIRED_TOP = {
    "timestamp":      _NUMBER,
    "route_id":       str,
    "operator_id":    str,
    "environment_id": str,
    "lidar_ref":      str,
    "collision":      bool,
    "reward_est":     _NUMBER,
    "priority":       _NUMBER,
}

POSE_KEYS = {"x": _NUMBER, "y": _NUMBER, "yaw": _NUMBER}
ODOM_KEYS = {"linear_x": _NUMBER, "angular_z": _NUMBER}
IMU_KEYS = {
    "roll":  _NUMBER,
    "pitch": _NUMBER,
    "yaw":   _NUMBER,
    "angular_velocity":   dict,
    "linear_acceleration": dict,
}
XYZ_KEYS = {"x": _NUMBER, "y": _NUMBER, "z": _NUMBER}
ACTION_KEYS = {"linear_x": _NUMBER, "angular_z": _NUMBER}


def _check_keys(d: Any, spec: dict, path: str, errors: list[str]) -> None:
    if not isinstance(d, dict):
        errors.append(f"{path}: expected dict, got {type(d).__name__}")
        return
    for key, typ in spec.items():
        if key not in d:
            errors.append(f"{path}.{key}: missing")
            continue
        if not isinstance(d[key], typ):
            errors.append(
                f"{path}.{key}: expected {typ}, got {type(d[key]).__name__}")


def validate_sample(sample: dict) -> list[str]:
    errors: list[str] = []
    _check_keys(sample, REQUIRED_TOP, "sample", errors)
    _check_keys(sample.get("pose", {}), POSE_KEYS, "sample.pose", errors)
    _check_keys(sample.get("odom", {}), ODOM_KEYS, "sample.odom", errors)
    imu = sample.get("imu", {})
    _check_keys(imu, IMU_KEYS, "sample.imu", errors)
    _check_keys(imu.get("angular_velocity", {}), XYZ_KEYS,
                "sample.imu.angular_velocity", errors)
    _check_keys(imu.get("linear_acceleration", {}), XYZ_KEYS,
                "sample.imu.linear_acceleration", errors)
    _check_keys(sample.get("action", {}), ACTION_KEYS,
                "sample.action", errors)
    return errors


def is_valid(sample: dict) -> bool:
    return not validate_sample(sample)
