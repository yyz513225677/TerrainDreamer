#!/usr/bin/env bash
# snapshot_gui.sh — capture Gazebo Sim + td_viewer window screenshots into
# paper/figures/ at the current moment.
#
# Requires:
#   - ImageMagick `import` (apt: imagemagick)
#   - wmctrl (apt: wmctrl)
#   - DISPLAY pointing at the X server where the GUIs are running
#
# Usage: ./scripts/snapshot_gui.sh [tag]
#   tag defaults to a YMD-HMS timestamp.
set -uo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
FIGS="$PROJECT/paper/figures"
mkdir -p "$FIGS"
DISPLAY="${DISPLAY:-:1}"
export DISPLAY

gz_id=$(wmctrl -l 2>/dev/null | awk '/Gazebo Sim/ {print $1; exit}')
vw_id=$(wmctrl -l 2>/dev/null | awk '/TerrainDreamer • Live Telemetry/ {print $1; exit}')

if [ -n "$gz_id" ]; then
  import -window "$gz_id" "$FIGS/gz_sim_$TAG.png"
  echo "saved $FIGS/gz_sim_$TAG.png"
else
  echo "Gazebo Sim window not found (is the sim running?)"
fi
if [ -n "$vw_id" ]; then
  import -window "$vw_id" "$FIGS/viewer_$TAG.png"
  echo "saved $FIGS/viewer_$TAG.png"
else
  echo "td_viewer window not found"
fi
