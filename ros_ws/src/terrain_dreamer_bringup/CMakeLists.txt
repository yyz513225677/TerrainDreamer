cmake_minimum_required(VERSION 3.0.2)
project(terrain_dreamer_bringup)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY launch worlds urdf models config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/generate_heightmaps.py
  scripts/generate_worlds.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
