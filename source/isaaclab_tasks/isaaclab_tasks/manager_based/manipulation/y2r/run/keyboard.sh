#!/bin/bash
# ==============================================================================
# Keyboard Debug - Palm Orientation Exploration
# ==============================================================================
# Usage:
#   ./y2r_sim/run/keyboard.sh                           # ur5e_leap (default, depth enabled)
#   ./y2r_sim/run/keyboard.sh --robot kuka_allegro      # kuka_allegro
#   ./y2r_sim/run/keyboard.sh --task <name>             # custom task layer
#
# Controls: WASDQE=move, ZXTGCV=rotate, L=reset, P=save depth, ESC=quit
# Palm orientation is printed every ~1 second.
# ==============================================================================

source "$(dirname "$0")/common.sh"

TASK_LAYER="base"
EXTRA_ARGS=("--enable_cameras" "--depth")

while [[ $# -gt 0 ]]; do
    case $1 in
        --robot)
            ROBOT="$2"
            TASK=$(resolve_robot_task "$ROBOT")
            shift 2
            ;;
        --task)
            TASK_LAYER="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

cd "$ISAACLAB_DIR"

echo "========================================"
echo "Keyboard Debug | Robot: $ROBOT | Task: $TASK_LAYER"
echo "========================================"
echo "Controls:"
echo "  W/S     - Move forward/backward (X)"
echo "  A/D     - Move left/right (Y)"
echo "  Q/E     - Move up/down (Z)"
echo "  Z/X     - Roll"
echo "  T/G     - Pitch"
echo "  C/V     - Yaw"
echo "  K       - Toggle gripper open/close"
echo "  L       - Reset keyboard deltas"
echo "  P       - Save wrist depth image"
echo "  ESC     - Quit"
echo "========================================"
echo "[Depth camera enabled — press P to save]"

Y2R_MODE=keyboard Y2R_TASK=$TASK_LAYER Y2R_ROBOT=$ROBOT ./isaaclab.sh -p \
    source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/scripts/keyboard_debug.py \
    --task "$TASK" \
    --livestream 2 \
    "${EXTRA_ARGS[@]}"
