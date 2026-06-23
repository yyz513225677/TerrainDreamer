from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

PKG = "lunar_south_pole_gazebo"


def _files(pattern: str) -> list:
    return [str(p) for p in Path(".").glob(pattern) if p.is_file()]


setup(
    name=PKG,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{PKG}"]),
        (f"share/{PKG}", ["package.xml"]),
        (f"share/{PKG}/launch", _files("launch/*.py")),
        (f"share/{PKG}/worlds", _files("worlds/*.sdf")),
        # Install the bridge + DEM YAMLs into the package share so
        # bridge.launch.py can resolve them via FindPackageShare.
        # The project's root config/ is at ../../../config from this
        # package's setup.py (pkg → src → ros2_ws → project root).
        (f"share/{PKG}/config",
         _files("../../../config/*.yaml")),
        (f"share/{PKG}/models/lunar_dem_terrain",
         _files("models/lunar_dem_terrain/*")),
        (f"share/{PKG}/models/lunar_rocks",
         _files("models/lunar_rocks/*")),
        (f"share/{PKG}/models/jackal_placeholder",
         _files("models/jackal_placeholder/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Maintainer",
    maintainer_email="yyz513225677@gmail.com",
    description="Lunar South Pole Gazebo Sim backend (stage 1).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"dem_metadata_node = {PKG}.dem_metadata_node:main",
            f"terrain_manager_node = {PKG}.terrain_manager_node:main",
            f"hazard_status_node = {PKG}.hazard_status_node:main",
            f"dreamer_interface_node = {PKG}.dreamer_interface_node:main",
            # Phase 2
            f"goal_marker_node = {PKG}.goal_marker_node:main",
            f"goal_status_node = {PKG}.goal_status_node:main",
            f"route_recorder_node = {PKG}.route_recorder_node:main",
            f"route_visualizer_node = {PKG}.route_visualizer_node:main",
            f"lunar_dashboard_bridge = {PKG}.lunar_dashboard_bridge:main",
            # Phase 3 — mission loop with random goals + flags
            f"mission_manager_node = {PKG}.mission_manager_node:main",
            f"fallback_nav_node = {PKG}.fallback_nav_node:main",
        ],
    },
)
