#!/bin/bash
# ==============================================================================
# Generate procedural shapes for Y2R trajectory task
# ==============================================================================
# Usage:
#   ./y2r_sim/run/generate_procedural.sh                    # Generate from config
#   ./y2r_sim/run/generate_procedural.sh --regenerate       # Force regeneration
#   ./y2r_sim/run/generate_procedural.sh -n 20 -s 42       # 20 shapes, seed 42
# ==============================================================================

source "$(dirname "$0")/common.sh"

cd "$ISAACLAB_DIR"

# generate_shapes.py lives in y2r/scripts/, one level up from run/
Y2R_DIR="$(dirname "$SCRIPT_DIR")"
./isaaclab.sh -p "$Y2R_DIR/scripts/generate_shapes.py" "$@"
