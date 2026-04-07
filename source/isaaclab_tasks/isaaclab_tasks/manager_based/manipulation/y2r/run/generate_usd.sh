#!/bin/bash
# ==============================================================================
# Test URDF → USD conversion for UR5e + LEAP Hand
# ==============================================================================
# Usage:
#   ./y2r_sim/run/generate_usd.sh            # Headless report only
#   ./y2r_sim/run/generate_usd.sh --view     # Open browser viewer at localhost:8211
# ==============================================================================

source "$(dirname "$0")/common.sh"

cd "$ISAACLAB_DIR"

./isaaclab.sh -p "$SCRIPT_DIR/generate_usd.py" --headless "$@"
