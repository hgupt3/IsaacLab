#!/bin/bash
# ==============================================================================
# Test URDF → USD conversion for Y2R robots
# ==============================================================================
# Usage:
#   ./y2r_sim/run/generate_usd.sh                                  # Headless report only
#   ./y2r_sim/run/generate_usd.sh --robot ur5e_gemini_wsg50 --view  # Browser viewer
# ==============================================================================

source "$(dirname "$0")/common.sh"

cd "$ISAACLAB_DIR"

./isaaclab.sh -p "$SCRIPT_DIR/generate_usd.py" --headless "$@"
