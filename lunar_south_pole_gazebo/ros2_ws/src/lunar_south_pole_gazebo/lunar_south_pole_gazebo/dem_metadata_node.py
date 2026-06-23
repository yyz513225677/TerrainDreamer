"""dem_metadata_node — publish the active tile's metadata once.

Reads the metadata YAML emitted by scripts/normalize_heightmap.py and
exposes it as a one-shot std_msgs/String (JSON payload) on
/dem/metadata so any downstream consumer can know what the world
"means" without parsing the world SDF.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import yaml


def load_metadata(path: Path) -> dict:
    """Pure function — load + validate. Importable by tests."""
    if not path.is_file():
        raise FileNotFoundError(f"metadata YAML not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    required = {
        "tile_width_m", "tile_height_m",
        "samples_x", "samples_y",
        "min_elevation_m", "max_elevation_m",
        "vertical_scale_m",
    }
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"metadata missing keys: {sorted(missing)}")
    return data


def main(args: Optional[list] = None) -> int:  # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except ImportError:
        print("[dem_metadata_node] rclpy not available — "
              "is the ROS 2 environment sourced?", file=sys.stderr)
        return 1

    rclpy.init(args=args)
    node = Node("dem_metadata_node")
    path_param = node.declare_parameter(
        "metadata_yaml", os.environ.get(
            "LUNAR_TILE_META",
            "data/metadata/shackleton_tile.yaml")).value
    pub = node.create_publisher(String, "/dem/metadata", 1)

    try:
        meta = load_metadata(Path(path_param))
    except (FileNotFoundError, ValueError) as exc:
        node.get_logger().error(str(exc))
        # Fall back to publishing an empty JSON so consumers see "no DEM".
        meta = {}

    msg = String()
    msg.data = json.dumps(meta)
    # Latch via QoS would be ideal; for stage-1 we just republish at 1 Hz.
    timer = node.create_timer(1.0, lambda: pub.publish(msg))
    node.get_logger().info(
        f"dem_metadata_node publishing /dem/metadata "
        f"(keys={list(meta.keys())})")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        timer.cancel()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
