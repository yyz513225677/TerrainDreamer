"""Tests for config/lunar_goals.yaml and the goal_marker_node loader."""
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve()
PKG_DIR = HERE.parent.parent
PROJECT_ROOT = HERE.parents[4]

sys.path.insert(0, str(PKG_DIR))
from lunar_south_pole_gazebo.goal_marker_node import load_goals  # noqa: E402


GOALS_YAML = PROJECT_ROOT / "config" / "lunar_goals.yaml"


def test_lunar_goals_yaml_exists():
    assert GOALS_YAML.is_file(), f"{GOALS_YAML} missing"


def test_lunar_goals_yaml_parses_via_loader():
    goals, defaults = load_goals(GOALS_YAML)
    assert len(goals) >= 5, "spec requires 5 example goals"
    expected_ids = {
        "goal_01_start_ridge", "goal_02_crater_rim",
        "goal_03_safe_basin", "goal_04_shadow_edge",
        "goal_05_return_base",
    }
    found = {g["goal_id"] for g in goals}
    assert expected_ids <= found, f"missing goal IDs: {expected_ids - found}"


def test_goal_required_fields():
    goals, _ = load_goals(GOALS_YAML)
    for g in goals:
        for k in ("goal_id", "name", "x", "y"):
            assert k in g
        # numeric coords
        for k in ("x", "y"):
            assert isinstance(g[k], (int, float))


def test_active_goal_id_resolves():
    data = yaml.safe_load(GOALS_YAML.read_text())
    if "active_goal_id" in data:
        ids = {g["goal_id"] for g in data["goals"]}
        assert data["active_goal_id"] in ids, \
            "active_goal_id must reference an existing goal"


def test_load_goals_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_goals(tmp_path / "nope.yaml")


def test_load_goals_missing_required_field_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({
        "goals": [{"goal_id": "g1", "name": "x"}],  # missing x,y
    }))
    with pytest.raises(ValueError):
        load_goals(bad)
