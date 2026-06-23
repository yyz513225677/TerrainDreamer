"""Recording schema tests.

Every line in a recording's JSONL file must conform to the RouteSample
schema (the wire shape used by route_recording_node). The seed demo
under data/demonstrations/ is the golden example.
"""
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent  # jackal_dreamer_dashboard/
SEED_JSONL = ROOT / "data" / "demonstrations" / "demo_seed_apennine_loop.jsonl"
SEED_ROUTE_META = ROOT / "data" / "routes" / "demo_seed_apennine_loop.json"
REPLAY_INDEX = ROOT / "data" / "replay_buffer" / "index.json"


# Required keys on every RouteSample record
REQUIRED_KEYS = {
    "timestamp", "route_id", "operator_id", "environment_id",
    "pose", "odom", "imu", "action", "collision", "reward_est", "priority",
}
POSE_KEYS = {"x", "y", "yaw"}
ODOM_KEYS = {"linear_x", "angular_z"}
IMU_KEYS = {"roll", "pitch", "yaw", "angular_velocity", "linear_acceleration"}
VEC3_KEYS = {"x", "y", "z"}
ACTION_KEYS = {"linear_x", "angular_z"}


def _load_jsonl(path: Path):
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def test_seed_jsonl_exists():
    assert SEED_JSONL.is_file(), \
        f"seed demonstration missing at {SEED_JSONL} — run the seed " \
        "generator before testing"


def test_every_record_has_required_keys():
    for lineno, rec in _load_jsonl(SEED_JSONL):
        missing = REQUIRED_KEYS - set(rec.keys())
        assert not missing, \
            f"line {lineno}: missing required keys {missing}"


def test_nested_shapes_correct():
    for lineno, rec in _load_jsonl(SEED_JSONL):
        assert POSE_KEYS <= set(rec["pose"]), f"line {lineno}: bad pose"
        assert ODOM_KEYS <= set(rec["odom"]), f"line {lineno}: bad odom"
        assert IMU_KEYS <= set(rec["imu"]), f"line {lineno}: bad imu"
        assert VEC3_KEYS <= set(rec["imu"]["angular_velocity"]), \
            f"line {lineno}: bad imu.angular_velocity"
        assert VEC3_KEYS <= set(rec["imu"]["linear_acceleration"]), \
            f"line {lineno}: bad imu.linear_acceleration"
        assert ACTION_KEYS <= set(rec["action"]), f"line {lineno}: bad action"


def test_types_are_numbers_where_expected():
    for lineno, rec in _load_jsonl(SEED_JSONL):
        for k in ("timestamp", "reward_est", "priority"):
            assert isinstance(rec[k], (int, float)), \
                f"line {lineno}: {k} must be numeric"
        for k in POSE_KEYS:
            assert isinstance(rec["pose"][k], (int, float)), \
                f"line {lineno}: pose.{k} must be numeric"
        assert isinstance(rec["collision"], bool), \
            f"line {lineno}: collision must be bool"


def test_route_id_consistent_within_file():
    seen = set()
    for _, rec in _load_jsonl(SEED_JSONL):
        seen.add(rec["route_id"])
    assert len(seen) == 1, \
        f"a recording must have one route_id; found {seen}"


def test_route_manifest_matches_jsonl():
    """The /data/routes/<route_id>.json manifest must match the JSONL."""
    assert SEED_ROUTE_META.is_file()
    meta = json.loads(SEED_ROUTE_META.read_text())
    samples = [r for _, r in _load_jsonl(SEED_JSONL)]
    assert meta["frame_count"] == len(samples), \
        "manifest frame_count must equal JSONL line count"
    assert meta["route_id"] == samples[0]["route_id"], \
        "manifest route_id must match samples"


def test_priority_one_demos_have_priority_field():
    for _, rec in _load_jsonl(SEED_JSONL):
        assert rec["priority"] == 1.0, \
            "seed demo is a human expert route — every sample must have " \
            "priority=1.0"


def test_replay_buffer_contains_seed_route():
    assert REPLAY_INDEX.is_file()
    buf = json.loads(REPLAY_INDEX.read_text())
    assert isinstance(buf, list)
    seed_entries = [e for e in buf if e["route_id"]
                    == "demo_seed_apennine_loop"]
    assert len(seed_entries) == 1, \
        "seed route must appear in replay_buffer/index.json"
    assert seed_entries[0]["priority"] == 1.0, \
        "expert demonstration must be priority=1.0"


if __name__ == "__main__":
    # Allow `python3 tests/test_recording_schema.py` for quick local runs
    pytest.main([__file__, "-q"])
