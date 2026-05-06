#!/bin/bash
# ==============================================================================
# Y2R Student Distillation Ablations: 3 conditions × 3 seeds = 9 sequential runs
# ==============================================================================
# Conditions:
#   baseline   - Hybrid (PPO + BC), depth + PT, current default settings
#   dagger     - Pure DAgger (50/50 BC, no PPO), lr 1e-4, no value distillation
#   no_depth   - Hybrid (same as baseline) but with the wrist depth camera
#                + encoder removed entirely (Y2R_MODE=distill_no_depth applies
#                configs/layers/no_depth.yaml on top of student.yaml)
#
# All 9 runs distill from the SAME teacher checkpoint (passed via --t_checkpoint).
#
# wandb layout (project = distillation_ablations):
#   group = condition (e.g. "baseline") -- wandb groups the 3 seeds together
#   name  = "<condition>_seed<N>"
#   tags  = [ablation, <condition>, seed<N>]
#
# Robustness:
#   - Per-run stdout/stderr is tee'd to logs/ablations_student/<run_name>.log
#   - Status manifest at logs/ablations_student/STATUS.tsv tracks progress
#   - A failing run does NOT abort the sweep; the next run starts anyway
#   - Re-running this script SKIPS runs already marked "done" (resume-safe)
#
# Usage:
#   ./y2r_sim/run/ablations_student.sh --t_checkpoint /path/to/teacher.pth
#   ./y2r_sim/run/ablations_student.sh --t_checkpoint t.pth --condition baseline
#   ./y2r_sim/run/ablations_student.sh --t_checkpoint t.pth --seeds 1,2
#   ./y2r_sim/run/ablations_student.sh --t_checkpoint t.pth --max_iterations 5000
#   ./y2r_sim/run/ablations_student.sh --t_checkpoint t.pth --force
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
MAX_ITERATIONS=10000           # Distillation iterations per run (~13 hr/run on RTX 5090)
WANDB_PROJECT="distillation_ablations"
CONDITIONS=("baseline" "dagger" "no_depth")
SEEDS=(1 2 3)
FORCE=0
TEACHER_CHECKPOINT=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --t_checkpoint)   TEACHER_CHECKPOINT="$2"; shift 2 ;;
        --condition)      CONDITIONS=("$2"); shift 2 ;;
        --seeds)          IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
        --max_iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        --project)        WANDB_PROJECT="$2"; shift 2 ;;
        --force)          FORCE=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$TEACHER_CHECKPOINT" ]; then
    echo "Error: --t_checkpoint <path/to/teacher.pth> is required."
    echo "All 9 distillation runs share the same teacher checkpoint."
    exit 1
fi
if [ ! -f "$TEACHER_CHECKPOINT" ]; then
    echo "Error: teacher checkpoint not found: $TEACHER_CHECKPOINT"
    exit 1
fi

sweep_init "ablations_student"

run_one() {
    local condition="$1" seed="$2"
    local student y2r_mode
    case "$condition" in
        baseline) student="pt";          y2r_mode="distill" ;;
        dagger)   student="pt_dagger";   y2r_mode="distill" ;;
        no_depth) student="pt_no_depth"; y2r_mode="distill_no_depth" ;;
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
    echo "  ABLATION: $run_name  (mode=$y2r_mode, student=$student, seed=$seed)"
    echo "  teacher: $TEACHER_CHECKPOINT"
    echo "  max_iterations=$MAX_ITERATIONS  project=$WANDB_PROJECT"
    echo "  log: $log_file"
    echo "=================================================================="
    echo ""
    sweep_set_status "$run_name" "running" ""

    # Run, tee output, capture exit code from the actual command (not tee).
    Y2R_MODE="$y2r_mode" "$SCRIPT_DIR/distill.sh" \
        --student "$student" \
        --t_checkpoint "$TEACHER_CHECKPOINT" \
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

sweep_print_header "Y2R Student Distillation Ablation Sweep"
echo "  Teacher:    $TEACHER_CHECKPOINT"
echo "=================================================================="

# Seed-major order: complete all conditions for seed N before moving to seed N+1.
# Surfaces a full one-seed comparison (baseline vs dagger vs no_depth) early.
for seed in "${SEEDS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        run_one "$condition" "$seed"
    done
done

sweep_print_final
