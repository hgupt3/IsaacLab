#!/bin/bash
# ==============================================================================
# Benchmark Y2R Env Step Timing
# ==============================================================================
# Profiles each section of env.step() with CUDA-synced timers.
# num_envs, decimation, etc. come from YAML config (base.yaml + mode layers).
#
# Usage:
#   ./y2r_sim/run/benchmark.sh                              # Teacher
#   ./y2r_sim/run/benchmark.sh --student                    # Student (distill mode)
#   ./y2r_sim/run/benchmark.sh --num_envs 8192              # Override num_envs
#   ./y2r_sim/run/benchmark.sh --torch_profile              # + Chrome trace
#   ./y2r_sim/run/benchmark.sh --warmup 100 --steps 500     # Custom step counts
#   ./y2r_sim/run/benchmark.sh --task cup                   # Task layer
# ==============================================================================

BENCHMARK_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$BENCHMARK_DIR/common.sh"

STUDENT=0
NUM_ENVS_ARG=""
WARMUP_STEPS=50
PROFILE_STEPS=200
TORCH_PROFILE=""
TASK_LAYER="base"
FORWARD_ARGS=()

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
        --student)
            STUDENT=1
            shift
            ;;
        --num_envs)
            NUM_ENVS_ARG="--num_envs $2"
            shift 2
            ;;
        --warmup)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --steps)
            PROFILE_STEPS="$2"
            shift 2
            ;;
        --torch_profile)
            TORCH_PROFILE="--torch_profile"
            shift
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

cd "$ISAACLAB_DIR"

# Student vs teacher: only difference is Y2R_MODE and camera flag.
# Y2R_MODE controls which config layers load (student.yaml sets
# use_student_mode: true and num_envs: 2048). The --agent entry point
# is only needed for hydra config loading — we step with random actions.
if [ "$STUDENT" = "1" ]; then
    MODE="distill"
    CAMERA_FLAG="--enable_cameras"
    AGENT_FLAG="--agent rl_games_student_cfg_entry_point"
    echo "========================================"
    echo "Benchmark: Student | Task: $TASK_LAYER"
    echo "========================================"
else
    MODE="train"
    CAMERA_FLAG=""
    AGENT_FLAG=""
    echo "========================================"
    echo "Benchmark: Teacher | Task: $TASK_LAYER"
    echo "========================================"
fi

Y2R_MODE=$MODE Y2R_TASK=$TASK_LAYER Y2R_ROBOT=$ROBOT \
    ./isaaclab.sh -p "$BENCHMARK_DIR/profile_step.py" \
    --task "$TASK" \
    --headless \
    $CAMERA_FLAG \
    $AGENT_FLAG \
    $NUM_ENVS_ARG \
    --warmup_steps "$WARMUP_STEPS" \
    --profile_steps "$PROFILE_STEPS" \
    $TORCH_PROFILE \
    "${FORWARD_ARGS[@]}"
