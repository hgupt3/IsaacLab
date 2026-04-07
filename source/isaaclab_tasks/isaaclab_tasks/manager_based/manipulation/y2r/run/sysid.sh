#!/bin/bash
# ==============================================================================
# SysID — Fit sim actuator parameters to real step response data
# ==============================================================================
# Usage:
#   ./y2r_sim/run/sysid.sh                        # fit hand (default, uses latest data)
#   ./y2r_sim/run/sysid.sh --arm                  # fit arm
#   ./y2r_sim/run/sysid.sh --hand --arm           # fit both
#   ./y2r_sim/run/sysid.sh --plot-only            # just plot current sim vs real
#   ./y2r_sim/run/sysid.sh --data path/to/file.npz  # explicit data file
#
# Flags:
#   --hand              Fit hand joints (default if no flag given)
#   --arm               Fit arm joints
#   --data <path>       .npz file(s) (default: latest in sysid_data/)
#   --plot-only         Skip optimization, just plot current sim vs real
#   --num_envs <N>      Override env count (default from sysid.yaml: 128)
#   --maxiter <N>       CMA-ES max iterations (default: 50)
# ==============================================================================

source "$(dirname "$0")/common.sh"

# Default data: latest hand .npz in sysid_data/
SYSID_DATA_DIR="$(dirname "$0")/../real_world_execution/sysid_data"
DATA_ARGS=()
HAS_DATA=0

for arg in "$@"; do
    if [ "$arg" = "--data" ]; then HAS_DATA=1; fi
done

if [ "$HAS_DATA" = "0" ]; then
    # Grab latest hand + arm npz files
    FILES=()
    LATEST_HAND=$(ls -t "$SYSID_DATA_DIR"/hand_*.npz 2>/dev/null | head -1)
    LATEST_ARM=$(ls -t "$SYSID_DATA_DIR"/arm_*.npz 2>/dev/null | head -1)
    [ -n "$LATEST_HAND" ] && FILES+=("$LATEST_HAND")
    [ -n "$LATEST_ARM" ] && FILES+=("$LATEST_ARM")

    if [ ${#FILES[@]} -eq 0 ]; then
        echo "Error: No *.npz found in $SYSID_DATA_DIR. Run sysid_collect.py first."
        exit 1
    fi
    echo "Using data: ${FILES[*]}"
    DATA_ARGS=(--data "${FILES[@]}")
fi

cd "$ISAACLAB_DIR"

Y2R_MODE=sysid Y2R_TASK=base Y2R_ROBOT=$ROBOT ./isaaclab.sh -p \
    source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/scripts/sysid_fit_isaac.py \
    --task "$TASK" \
    --livestream 2 \
    --enable_cameras \
    "${DATA_ARGS[@]}" \
    "$@"
