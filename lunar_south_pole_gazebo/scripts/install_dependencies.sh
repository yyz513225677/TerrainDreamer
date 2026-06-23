#!/usr/bin/env bash
# Install system + Python dependencies for lunar_south_pole_gazebo.
# Uses apt-get where possible — none of the Python pieces here are
# project-specific; we just route the user to existing packages.

set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-jazzy}"

echo "[install] ROS distro target: ${ROS_DISTRO}"
echo "[install] (override with: ROS_DISTRO=humble bash $0)"
echo

if ! command -v sudo >/dev/null 2>&1; then
  echo "[install] 'sudo' not on PATH; run this script as root or install sudo." >&2
  exit 1
fi

APT_PKGS=(
  gdal-bin
  python3-gdal
  python3-numpy
  python3-yaml
  python3-pil
  python3-rasterio
  "ros-${ROS_DISTRO}-ros-gz-bridge"
)

echo "[install] running: sudo apt-get update"
sudo apt-get update -qq

echo "[install] installing: ${APT_PKGS[*]}"
# Don't fail the whole script if a single package isn't published
# for this distro — report what's missing at the end instead.
MISSING=()
for pkg in "${APT_PKGS[@]}"; do
  if ! sudo apt-get install -y --no-install-recommends "$pkg" 2>/dev/null; then
    echo "[install] WARN — could not install $pkg via apt"
    MISSING+=("$pkg")
  fi
done

echo
echo "[install] verifying tools on PATH …"
for cmd in gdalinfo gdal_translate; do
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "  [ok] $cmd → $(command -v $cmd)"
  else
    echo "  [MISSING] $cmd"
    MISSING+=("$cmd")
  fi
done

echo
echo "[install] verifying Python modules …"
for mod in numpy yaml PIL; do
  if python3 -c "import $mod" 2>/dev/null; then
    echo "  [ok] python3 -c 'import $mod'"
  else
    echo "  [MISSING] python3 import $mod"
    MISSING+=("python3-$mod")
  fi
done

# rasterio is preferred but optional — the scripts can fall back to GDAL
if python3 -c "import rasterio" 2>/dev/null; then
  echo "  [ok] python3 -c 'import rasterio'"
else
  echo "  [optional] rasterio not importable — scripts will fall back to osgeo.gdal"
fi

# osgeo (GDAL bindings) — needed for the rasterio fallback path
if python3 -c "from osgeo import gdal" 2>/dev/null; then
  echo "  [ok] python3 -c 'from osgeo import gdal'"
else
  echo "  [MISSING] python3 'from osgeo import gdal' — install python3-gdal"
  MISSING+=("python3-gdal")
fi

echo
if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "[install] FINISHED with missing items:"
  printf '  - %s\n' "${MISSING[@]}"
  echo
  echo "  Fix manually before running the pipeline."
  exit 2
fi

echo "[install] all dependencies satisfied."
