#!/usr/bin/env bash
# capture_gz_screenshot.sh — attach a GUI client to the running headless
# gz-sim server and grab a screenshot of the currently-loaded world.
#
# Usage: bash scripts/capture_gz_screenshot.sh [out_basename]
# Default output: paper/figures/screenshot_gz_sim_<terrain>.png
#
# Notes
#   * Does NOT touch the trainer. The GUI client is a separate read-only
#     view of the same gz transport partition.
#   * Bypasses snap-wrapped /usr/bin/gz (libpthread symbol-lookup error)
#     by sourcing the running trainer's environment block from /proc.
#   * Auto-times out if the GUI fails to show a window.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_BASE="${1:-screenshot_gz_sim_extreme}"
OUT="$PROJECT_ROOT/paper/figures/${OUT_BASE}.png"
SETTLE_SEC="${SETTLE_SEC:-30}"

log() { printf '\033[1;36m[capture %s]\033[0m %s\n' "$(date +%T)" "$*"; }

# 1. Locate the headless gz-sim server (the trainer's child).
SERVER_PID=$(pgrep -f "gz sim .*-s --headless-rendering" | head -1)
if [ -z "$SERVER_PID" ]; then
    log "no running headless gz-sim server found — start training first."
    exit 1
fi
log "found gz-sim server PID=$SERVER_PID"

# 2. Replay its environment + force a working DISPLAY so the GUI client
#    discovers the same world and ignores the snap-injected libs.
ENV_FILE="/tmp/gz_server_env_$$.sh"
{
    echo "unset \$(env | cut -d= -f1)"
    tr '\0' '\n' < "/proc/$SERVER_PID/environ" \
        | while IFS='=' read -r k v; do
            [ -z "$k" ] && continue
            printf 'export %s=%q\n' "$k" "$v"
        done
    echo "export DISPLAY=\${DISPLAY:-:1}"
    echo "export XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}"
    # Strip /snap from PATH so /usr/bin/gz can't shadow vendored binary.
    echo 'export PATH=$(printf "%s" "$PATH" | tr ":" "\n" | grep -v "^/snap/" | tr "\n" ":" | sed "s/:$//")'
} > "$ENV_FILE"

# 3. Launch the GUI client (gz_tools_vendor, forced to gz-sim 8).
log "spawning gz sim -g ..."
bash -c "source $ENV_FILE && \
         ruby /opt/ros/jazzy/opt/gz_tools_vendor/bin/gz sim -g --force-version 8" \
    > /tmp/gz_gui_client.log 2>&1 &
GUI_PID=$!
log "GUI client PID=$GUI_PID; waiting up to ${SETTLE_SEC}s for window ..."

# 4. Poll for the Gazebo Sim window to appear.
WID=""
for _ in $(seq 1 "$SETTLE_SEC"); do
    sleep 1
    WID=$(DISPLAY=:1 wmctrl -l 2>/dev/null \
          | awk '/Gazebo Sim/{print $1; exit}')
    [ -n "$WID" ] && break
done

if [ -z "$WID" ]; then
    log "no Gazebo Sim window — check /tmp/gz_gui_client.log."
    tail -10 /tmp/gz_gui_client.log
    kill -9 "$GUI_PID" 2>/dev/null
    rm -f "$ENV_FILE"
    exit 1
fi
log "found window $WID; settling 5 s for scene render ..."
DISPLAY=:1 wmctrl -i -a "$WID" 2>/dev/null
sleep 5

# 5. Capture.
DISPLAY=:1 import -window "$WID" "$OUT"
log "saved $OUT"
ls -la "$OUT"

# 6. Kill the GUI client (server unaffected).
log "closing GUI client ..."
kill -INT "$GUI_PID" 2>/dev/null
sleep 2
pkill -9 -f "ruby .*gz sim -g --force-version 8" 2>/dev/null
rm -f "$ENV_FILE"
log "done."
