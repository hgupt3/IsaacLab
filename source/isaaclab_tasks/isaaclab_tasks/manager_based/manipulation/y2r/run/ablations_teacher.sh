#!/bin/bash
# ==============================================================================
# Y2R Teacher Ablations: 3 conditions × 3 seeds = 9 sequential runs
# ==============================================================================
# Conditions:
#   pt              - Point Transformer (full model)
#   mlp             - Default MLP (training hyperparams matched to PT)
#   pt_no_eigen     - Point Transformer, no eigen grasps (raw 16D hand control)
#
# wandb layout (project = trajectory_ablations):
#   group = condition (e.g. "pt") -- wandb groups the 3 seeds together
#   name  = "<condition>_seed<N>"
#   tags  = [ablation, <condition>, seed<N>]
#
# Robustness:
#   - Per-run stdout/stderr is tee'd to logs/ablations/<run_name>.log
#   - Status manifest at logs/ablations/STATUS.tsv tracks pending/running/done/failed
#   - A failing run does NOT abort the sweep; the next run starts anyway
#   - Re-running this script SKIPS runs already marked "done" (resume-safe)
#
# Usage:
#   ./y2r_sim/run/ablations_teacher.sh                        # all 9 sequentially
#   ./y2r_sim/run/ablations_teacher.sh --condition pt         # only PT seeds
#   ./y2r_sim/run/ablations_teacher.sh --seeds 1,2            # only seeds 1 and 2
#   ./y2r_sim/run/ablations_teacher.sh --max_iterations 6250  # override step count
#   ./y2r_sim/run/ablations_teacher.sh --force                # re-run even if marked done
# ==============================================================================

# NOTE: deliberately NOT using `set -e` at top level — one bad run shouldn't
# abort the rest of the sweep. We capture exit codes explicitly per run.
set -uo pipefail

# common.sh resolves REPO_ROOT (parent containing .git), ISAACLAB_DIR, Y2R_DATA_ROOT.
# Manual ../.. traversal would land in .../manager_based/manipulation — wrong.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/sweep_lib.sh"

# Defaults
MAX_ITERATIONS=3125          # ≈ 100k common_step_counter (3125 * horizon_length=32)
WANDB_PROJECT="trajectory_ablations"
CONDITIONS=("pt" "mlp" "pt_no_eigen")
SEEDS=(1 2 3)
FORCE=0

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --condition)      CONDITIONS=("$2"); shift 2 ;;
        --seeds)          IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
        --max_iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        --project)        WANDB_PROJECT="$2"; shift 2 ;;
        --force)          FORCE=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

sweep_init "ablations"

run_one() {
    local condition="$1" seed="$2"
    local agent y2r_mode
    case "$condition" in
        pt)          agent="pt";  y2r_mode="train" ;;
        mlp)         agent="mlp"; y2r_mode="train" ;;
        pt_no_eigen) agent="pt";  y2r_mode="train_no_eigen" ;;
        *) echo "Unknown condition: $condition"; return 1 ;;
    esac

    local run_name="${condition}_seed${seed}"
    local log_file="$LOG_DIR/$run_name.log"

    if [ "$FORCE" -eq 0 ] && sweep_is_done "$run_name"; then
        echo ">>> SKIP $run_name (already done; pass --force to re-run)"
        return 0
    fi

    echo ""
    echo "=================================================================="
    echo "  ABLATION: $run_name  (mode=$y2r_mode, agent=$agent, seed=$seed)"
    echo "  max_iterations=$MAX_ITERATIONS  project=$WANDB_PROJECT"
    echo "  log: $log_file"
    echo "=================================================================="
    echo ""
    sweep_set_status "$run_name" "running" ""

    # Run, tee output, capture exit code from the actual command (not tee).
    Y2R_MODE="$y2r_mode" "$SCRIPT_DIR/train.sh" \
        --agent "$agent" \
        --seed "$seed" \
        --max_iterations "$MAX_ITERATIONS" \
        --wandb-project-name "$WANDB_PROJECT" \
        --wandb-group "$condition" \
        --wandb-name "$run_name" \
        --wandb-tags "ablation,$condition,seed$seed" \
        2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    if [ "$exit_code" -eq 0 ]; then
        sweep_set_status "$run_name" "done" "$exit_code"
        echo ">>> OK $run_name (exit 0)"
    else
        sweep_set_status "$run_name" "failed" "$exit_code"
        echo ">>> FAIL $run_name (exit $exit_code) — continuing with next run"
    fi
    return 0
}

sweep_print_header "Y2R Teacher Ablation Sweep"

# Seed-major order: complete all conditions for seed N before moving to seed N+1.
# This way you get a full one-seed comparison (pt vs mlp vs pt_no_eigen) early,
# instead of waiting for all PT seeds to finish before any MLP seed starts.
for seed in "${SEEDS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        run_one "$condition" "$seed"
    done
done

sweep_print_final
