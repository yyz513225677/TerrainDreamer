"""spawn_jackal.launch.py — spawn the rover into a running Gazebo Sim.

Prefers the official Clearpath Jackal description if `jackal_description`
is installed. Falls back to the project's placeholder SDF otherwise.

Safe-Z spawn: if `z` is left at the sentinel `auto`, the launch reads
`data/metadata/south_pole_tile.yaml`'s `vertical_scale_m` and adds
`z_safe_above_dem` (default 2.0 m). This puts the rover comfortably
above the highest point of the heightmap; lunar gravity (1.62 m/s²)
makes the drop gentle.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PKG = "lunar_south_pole_gazebo"


def _have_official_jackal() -> bool:
    try:
        from ament_index_python.packages import get_package_share_directory
        get_package_share_directory("jackal_description")
        return True
    except Exception:
        return False


def _read_displayed_vscale(terrain_sdf: Optional[Path],
                           metadata_yaml: Optional[Path]) -> float:
    """Return the Z component of the terrain's *displayed* size.

    Phase 1 may rescale the heightmap when shipping the rendered world
    (e.g. 10 km × 10 km × 354 m instead of the full 200 km × 200 km ×
    7088 m mosaic). Reading the YAML's `vertical_scale_m` alone would
    overestimate the spawn height by 20× in that case and drop the
    rover from kilometres up.

    Source of truth: the `<size>` tag inside
    `models/lunar_dem_terrain/model.sdf`. The metadata YAML's
    `vertical_scale_m` is used as a fallback.
    """
    if terrain_sdf is not None and terrain_sdf.is_file():
        try:
            import xml.etree.ElementTree as ET
            root = ET.parse(terrain_sdf).getroot()
            for size_elem in root.iter("size"):
                parts = (size_elem.text or "").split()
                if len(parts) == 3:
                    return float(parts[2])
        except Exception:
            pass
    if metadata_yaml is not None and metadata_yaml.is_file():
        try:
            import yaml
            data = yaml.safe_load(metadata_yaml.read_text()) or {}
            return float(data.get("vertical_scale_m", 0.0))
        except Exception:
            pass
    return 0.0


def _launch_setup(context, *args, **kwargs):
    from ament_index_python.packages import get_package_share_directory

    cfg = lambda k: LaunchConfiguration(k).perform(context)
    x_s = cfg("x"); y_s = cfg("y"); z_s = cfg("z")
    roll = cfg("roll"); pitch = cfg("pitch"); yaw = cfg("yaw")
    robot_name = cfg("robot_name")
    use_official_jackal = cfg("use_official_jackal").lower() in {"true", "1"}
    z_safe_above_dem = float(cfg("z_safe_above_dem"))

    # Resolve the model SDF + metadata YAML
    pkg_share = Path(get_package_share_directory(PKG))
    placeholder_sdf = pkg_share / "models" / "jackal_placeholder" / "model.sdf"

    metadata_yaml = None
    # Project root sits at parents[3] from share/<pkg>/ in an installed layout.
    # Try both source-tree and CWD paths so the launch works either way.
    for candidate in (
        pkg_share.parents[3] / "data" / "metadata" / "south_pole_tile.yaml"
        if len(pkg_share.parents) >= 4 else None,
        Path.cwd() / "data" / "metadata" / "south_pole_tile.yaml",
        Path(os.environ.get("LUNAR_TILE_META", "")) if
        os.environ.get("LUNAR_TILE_META") else None,
    ):
        if candidate is not None and candidate.is_file():
            metadata_yaml = candidate
            break

    if z_s.strip().lower() == "auto":
        terrain_sdf = (pkg_share / "models" / "lunar_dem_terrain"
                       / "model.sdf")
        vscale = _read_displayed_vscale(terrain_sdf, metadata_yaml)
        # Sanity cap: if the terrain SDF parse failed AND the YAML is
        # wildly large (e.g. unscaled 7 km vscale on a rescaled tile),
        # cap spawn z to a value the rover can survive falling from.
        Z_MAX = 1000.0
        if vscale > Z_MAX:
            print(f"[spawn_jackal] WARN: vscale={vscale} > {Z_MAX} — "
                  f"capping (likely Phase-1 scaled-down terrain).",
                  flush=True)
            vscale = Z_MAX
        z_resolved = vscale + z_safe_above_dem
        print(f"[spawn_jackal] safe-Z: displayed_vscale={vscale} + "
              f"z_safe_above_dem={z_safe_above_dem} → z={z_resolved}",
              flush=True)
    else:
        z_resolved = float(z_s)

    sdf_path = str(placeholder_sdf)
    if use_official_jackal and _have_official_jackal():
        # TODO(stage-3): integrate Clearpath jackal_description when it
        # ships for Jazzy. Until then we still spawn the placeholder.
        print("[spawn_jackal] use_official_jackal=true but the "
              "Clearpath stack is not yet wired — spawning placeholder.",
              flush=True)

    print(f"[spawn_jackal] spawning {sdf_path} as '{robot_name}' at "
          f"({x_s}, {y_s}, {z_resolved}) yaw={yaw}", flush=True)

    return [
        Node(
            package="ros_gz_sim",
            executable="create",
            output="screen",
            arguments=[
                "-file", sdf_path,
                "-name", robot_name,
                "-x", x_s, "-y", y_s, "-z", str(z_resolved),
                "-R", roll, "-P", pitch, "-Y", yaw,
            ],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("z", default_value="auto",
                              description="explicit Z or 'auto' to derive "
                                          "from tile metadata"),
        DeclareLaunchArgument("z_safe_above_dem", default_value="2.0"),
        DeclareLaunchArgument("roll", default_value="0.0"),
        DeclareLaunchArgument("pitch", default_value="0.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("robot_name", default_value="lunar_jackal"),
        DeclareLaunchArgument("use_official_jackal", default_value="false"),
        # The following topic-name args are documented for downstream
        # consumers but the rover SDF bakes the gz-side topics in; the
        # bridge YAML maps them to these ROS names.
        DeclareLaunchArgument("cmd_vel_topic",
                              default_value="/lunar_jackal/cmd_vel"),
        DeclareLaunchArgument("scan_topic",
                              default_value="/lunar_jackal/scan"),
        DeclareLaunchArgument("imu_topic",
                              default_value="/lunar_jackal/imu"),
        DeclareLaunchArgument("odom_topic",
                              default_value="/lunar_jackal/odom"),
        OpaqueFunction(function=_launch_setup),
    ])
