#!/usr/bin/env bash
# run_baseline.sh — launch TerrainDreamer training in baseline modes for the paper.
#
# Baselines (set via $1):
#   crater       — full TerrainDreamer (default): demo-anchored + hierarchical + memory
#   vanilla      — vanilla DreamerV3 (no BC, no hierarchical, no memory)
#   no_demo      — ablation C1: drop demonstration-anchored regime
#   no_hier      — ablation C2: drop hierarchical sub-goal action
#   no_memory    — ablation C3: drop visited-ground memory channel
#
# Usage:
#   ./scripts/run_baseline.sh vanilla
#   TERRAIN=landscape ./scripts/run_baseline.sh crater
#   SEED=1337 ./scripts/run_baseline.sh no_hier
#
# Each invocation does a full cold restart via clean_and_train.sh, so
# previous trainers / sims are torn down first.

set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BASELINE="${1:-crater}"
shift || true

# Common defaults — terrain + headless. Override via env if needed.
export TERRAIN="${TERRAIN:-rugged}"
export HEADLESS="${HEADLESS:-1}"

case "$BASELINE" in
  crater)
    # Full TerrainDreamer: nothing to override.
    export BASE_POLICY_MODE="${BASE_POLICY_MODE:-reactive}"
    export USE_BASE_POLICY="${USE_BASE_POLICY:-1}"
    export USE_HIERARCHICAL_ACTION="${USE_HIERARCHICAL_ACTION:-1}"
    export BC_WEIGHT="${BC_WEIGHT:-2.0}"
    export DREAMER_ACTOR_WEIGHT="${DREAMER_ACTOR_WEIGHT:-0.3}"
    LABEL="crater_full"
    ;;
  vanilla)
    # Vanilla DreamerV3: no BC, flat action, no memory, no demos.
    export USE_BASE_POLICY=0
    export USE_HIERARCHICAL_ACTION=0
    export BC_WEIGHT=0
    export DREAMER_ACTOR_WEIGHT=1.0
    export BASE_POLICY_MODE=simple   # ignored when USE_BASE_POLICY=0
    LABEL="vanilla_dreamerv3"
    ;;
  no_demo)
    # Ablation C1: keep hierarchical + memory but drop demo-anchored regime.
    export USE_BASE_POLICY=0
    export BC_WEIGHT=0
    export DREAMER_ACTOR_WEIGHT=1.0
    export USE_HIERARCHICAL_ACTION=1
    export BASE_POLICY_MODE=simple
    LABEL="no_demo_anchor"
    ;;
  no_hier)
    # Ablation C2: drop hierarchical sub-goal; keep BC + demo + memory.
    export USE_HIERARCHICAL_ACTION=0
    export USE_BASE_POLICY=1
    export BC_WEIGHT="${BC_WEIGHT:-2.0}"
    export DREAMER_ACTOR_WEIGHT="${DREAMER_ACTOR_WEIGHT:-0.3}"
    export BASE_POLICY_MODE=reactive
    LABEL="no_hierarchical"
    ;;
  no_memory)
    # Ablation C3: keep everything but force base-policy mode without memory.
    export USE_BASE_POLICY=1
    export USE_HIERARCHICAL_ACTION=1
    export BC_WEIGHT="${BC_WEIGHT:-2.0}"
    export DREAMER_ACTOR_WEIGHT="${DREAMER_ACTOR_WEIGHT:-0.3}"
    export BASE_POLICY_MODE=reactive   # memory only kicks in with mode=memory
    LABEL="no_memory_channel"
    ;;
  *)
    echo "ERROR: unknown baseline '$BASELINE'" >&2
    echo "Available: crater | vanilla | no_demo | no_hier | no_memory" >&2
    exit 1
    ;;
esac

echo "================================================================"
echo "Baseline   : $LABEL"
echo "Terrain    : $TERRAIN"
echo "Headless   : $HEADLESS"
echo "USE_BASE_POLICY        = $USE_BASE_POLICY"
echo "USE_HIERARCHICAL_ACTION = $USE_HIERARCHICAL_ACTION"
echo "BASE_POLICY_MODE       = $BASE_POLICY_MODE"
echo "BC_WEIGHT              = $BC_WEIGHT"
echo "DREAMER_ACTOR_WEIGHT   = $DREAMER_ACTOR_WEIGHT"
echo "================================================================"

# Tag the next iter dir for analysis later.
LABEL_FILE="$PROJECT_ROOT/experiments/crater/.next_label"
echo "$LABEL" > "$LABEL_FILE"

exec ./scripts/clean_and_train.sh
