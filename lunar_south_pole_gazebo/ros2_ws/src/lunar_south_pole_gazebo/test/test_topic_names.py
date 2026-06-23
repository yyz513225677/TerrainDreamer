"""Contract test: the topic names declared in bridge YAMLs match
docs/jackal_sensor_goal_camera_plan.md §2.

If anyone renames a topic in the SDF or bridge without updating this
test or the design doc, this test fails — preventing silent drift.
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[4]


def _topic_set(path: Path) -> set:
    data = yaml.safe_load(path.read_text())
    assert isinstance(data, list)
    return {e["ros_topic_name"] for e in data}


# Required Phase-2 topic set per the spec.
PHASE2_REQUIRED = {
    "/clock", "/tf",
    "/lunar_jackal/scan", "/lunar_jackal/imu", "/lunar_jackal/odom",
    "/lunar_jackal/cmd_vel",
    "/lunar_jackal/chase_camera/image",
    "/lunar_jackal/chase_camera/camera_info",
}
PHASE1_ALIASES = {"/scan", "/imu", "/odom", "/cmd_vel"}


def test_bridge_jazzy_has_phase2_topics():
    p = PROJECT_ROOT / "config" / "bridge_jazzy.yaml"
    topics = _topic_set(p)
    missing = PHASE2_REQUIRED - topics
    assert not missing, f"bridge_jazzy.yaml missing topics: {missing}"


def test_bridge_jazzy_keeps_phase1_aliases():
    p = PROJECT_ROOT / "config" / "bridge_jazzy.yaml"
    topics = _topic_set(p)
    missing = PHASE1_ALIASES - topics
    assert not missing, \
        f"bridge_jazzy.yaml dropped Phase-1 aliases: {missing}"


def test_bridge_humble_has_phase2_topics():
    p = PROJECT_ROOT / "config" / "bridge_humble.yaml"
    topics = _topic_set(p)
    missing = PHASE2_REQUIRED - topics
    assert not missing, f"bridge_humble.yaml missing topics: {missing}"


def test_phase2_plan_mentions_each_topic():
    plan = (PROJECT_ROOT / "docs"
            / "jackal_sensor_goal_camera_plan.md").read_text()
    for t in PHASE2_REQUIRED:
        assert t in plan, f"plan should mention {t}"


def test_camera_info_maps_to_camera_info_gz_topic():
    """Regression test: an earlier draft had the camera_info entry's
    gz_topic_name accidentally set to the image topic. The two must be
    distinct gz topics."""
    for fname in ("bridge_jazzy.yaml", "bridge_humble.yaml"):
        data = yaml.safe_load((PROJECT_ROOT / "config" / fname).read_text())
        info = [e for e in data
                if e["ros_topic_name"]
                == "/lunar_jackal/chase_camera/camera_info"]
        assert info, f"{fname} missing camera_info entry"
        assert info[0]["gz_topic_name"].endswith("/camera_info"), \
            f"{fname}: camera_info gz topic must end in /camera_info, " \
            f"got {info[0]['gz_topic_name']}"
