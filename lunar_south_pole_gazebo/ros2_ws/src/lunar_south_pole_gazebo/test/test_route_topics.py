"""Tests that the bridge YAML's topic set matches the design contract
documented in docs/design.md §4 and README.md.

This is a *contract test* — if anyone edits the bridge config without
also editing the design doc, this test fails.
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[4]  # lunar_south_pole_gazebo/ project root


def _bridge_topics(path: Path) -> set:
    data = yaml.safe_load(path.read_text())
    assert isinstance(data, list), f"{path} must be a YAML list"
    return {entry["ros_topic_name"] for entry in data}


# Topic set published as Gz→ROS or ROS→Gz. Internal ROS-only topics
# (/dashboard/reset, /dashboard/estop, /collision_status) do NOT
# travel via ros_gz_bridge.
EXPECTED_BRIDGE_TOPICS = {
    "/clock", "/scan", "/imu", "/odom", "/tf", "/cmd_vel",
}


def test_bridge_jazzy_topic_set():
    # Phase-2 added /lunar_jackal/* topics; Phase-1 aliases must remain.
    p = PROJECT_ROOT / "config" / "bridge_jazzy.yaml"
    actual = _bridge_topics(p)
    missing = EXPECTED_BRIDGE_TOPICS - actual
    assert not missing, f"bridge_jazzy.yaml lost Phase-1 topics: {missing}"


def test_bridge_humble_topic_set():
    p = PROJECT_ROOT / "config" / "bridge_humble.yaml"
    actual = _bridge_topics(p)
    missing = EXPECTED_BRIDGE_TOPICS - actual
    assert not missing, f"bridge_humble.yaml lost Phase-1 topics: {missing}"


def test_design_doc_mentions_each_topic():
    design = (PROJECT_ROOT / "docs" / "design.md").read_text()
    for t in EXPECTED_BRIDGE_TOPICS:
        assert t in design, f"docs/design.md should mention {t}"


def test_readme_mentions_each_topic():
    readme = (PROJECT_ROOT / "README.md").read_text()
    for t in EXPECTED_BRIDGE_TOPICS:
        assert t in readme, f"README.md should mention {t}"


def test_no_invented_shackleton_coords_in_repo():
    """No file outside the docs tree may bake in lat/lon Shackleton coords."""
    forbidden_substrings = []  # add specific known-bad numbers here when seen
    # This guard test runs against the whole project tree.
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        # docs/qgis_workflow.md and docs/dem_source_notes.md are
        # allowed to discuss coords abstractly.
        if "docs" in path.parts:
            continue
        if path.suffix in {".png", ".jpg", ".tif"}:
            continue
        text = path.read_text(errors="ignore")
        for bad in forbidden_substrings:
            assert bad not in text, f"{path} contains forbidden coord '{bad}'"
