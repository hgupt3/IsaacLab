#!/bin/bash
# ==============================================================================
# Y2R Common Script Helpers
# ==============================================================================
# Source this file in other scripts: source "$(dirname "$0")/common.sh"
# ==============================================================================

# Dynamically determine paths relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Walk up from script location to find the repo root (contains .git directory)
_dir="$SCRIPT_DIR"
while [ "$_dir" != "/" ]; do
    if [ -d "$_dir/.git" ]; then break; fi
    _dir="$(dirname "$_dir")"
done
REPO_ROOT="$_dir"
ISAACLAB_DIR="$REPO_ROOT/IsaacLab"

# Y2R_DATA_ROOT controls where all large outputs go; defaults to repo root
Y2R_DATA_ROOT="${Y2R_DATA_ROOT:-$REPO_ROOT}"

# CUDA allocator default: reduce fragmentation-related OOMs.
# Respect user overrides and only inject expandable_segments if missing.
if [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
elif [[ ",${PYTORCH_CUDA_ALLOC_CONF}," != *",expandable_segments:"* ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF},expandable_segments:True"
fi

# ==============================================================================
# Robot Selection
# ==============================================================================
# Default robot. Override per-invocation with --robot <alias>.
# Available: ur5e_leap, kuka_allegro
# ==============================================================================
ROBOT="${ROBOT:-ur5e_leap}"

# ==============================================================================
# Multi-GPU Support
# ==============================================================================
# Usage: --multi_gpu 2,3     (use GPUs 2 and 3)
#        --multi_gpu 0,1,2,3 (use all 4 GPUs)
#        --multi_gpu 2       (use 2 GPUs starting from 0)
#
# Note: CUDA_VISIBLE_DEVICES is NOT set for multi-GPU because Isaac Sim's
# Vulkan renderer needs to see all physical GPUs. GPU selection is handled
# by torchrun LOCAL_RANK + the GPU_OFFSET environment variable.
# ==============================================================================
MULTI_GPU=""
NUM_GPUS=0
GPU_OFFSET=0
MASTER_PORT=29500

parse_multi_gpu() {
    local gpu_list="$1"
    MULTI_GPU="$gpu_list"

    # Check if it's just a count (single number, no commas)
    if [[ "$gpu_list" =~ ^[0-9]+$ ]] && [ "$gpu_list" -le 8 ] && [ "$gpu_list" -gt 1 ]; then
        # Just a count, e.g. --multi_gpu 4
        NUM_GPUS="$gpu_list"
        GPU_OFFSET=0
    else
        # Comma-separated list, e.g. --multi_gpu 2,3
        NUM_GPUS=$(echo "$gpu_list" | awk -F',' '{print NF}')
        # Use first GPU in list as offset (assumes contiguous)
        GPU_OFFSET=$(echo "$gpu_list" | cut -d',' -f1)
    fi

    # Derive unique master port from GPU offset to allow multiple torchrun instances
    MASTER_PORT=$((29500 + GPU_OFFSET))

    export Y2R_GPU_OFFSET="$GPU_OFFSET"
    echo "========================================"
    echo "Multi-GPU: $NUM_GPUS GPUs (offset=$GPU_OFFSET, port=$MASTER_PORT)"
    echo "========================================"
}

resolve_robot_task() {
    case "$1" in
        ur5e_leap)    echo "Isaac-Trajectory-UR5e-Leap-v0" ;;
        kuka_allegro) echo "Isaac-Trajectory-Kuka-Allegro-v0" ;;
        *)
            echo "Error: Unknown robot '$1'" >&2
            echo "Available robots: ur5e_leap, kuka_allegro" >&2
            exit 1
            ;;
    esac
}

TASK=$(resolve_robot_task "$ROBOT")

# Export robot selection for Python config loader
export Y2R_ROBOT="$ROBOT"

# ==============================================================================
# Agent Configuration
# ==============================================================================
# Maps short aliases to entry points and log directories
# Format: AGENT_<alias>_ENTRY and AGENT_<alias>_LOG
# ==============================================================================

# Default MLP agent
AGENT_mlp_ENTRY="rl_games_cfg_entry_point"
AGENT_mlp_LOG="trajectory"

# PointNet with T-Net
AGENT_tnet_ENTRY="rl_games_pointnet_tnet_cfg_entry_point"
AGENT_tnet_LOG="trajectory_pointnet_tnet"

# Point Transformer
AGENT_pt_ENTRY="rl_games_point_transformer_cfg_entry_point"
AGENT_pt_LOG="trajectory_point_transformer"

# Resolve agent alias to entry point
# Usage: ENTRY_POINT=$(resolve_agent_entry "tnet")
resolve_agent_entry() {
    local alias="$1"
    local var="AGENT_${alias}_ENTRY"
    echo "${!var}"
}

# Resolve agent alias to log directory
# Usage: LOG_DIR=$(resolve_agent_log "tnet")
resolve_agent_log() {
    local alias="$1"
    local var="AGENT_${alias}_LOG"
    echo "${!var}"
}

# Find the latest checkpoint in a given log directory
# Usage: CHECKPOINT=$(find_latest "trajectory")
find_latest() {
    local dir="$1"
    find "$Y2R_DATA_ROOT/IsaacLab/logs/rl_games/$dir" -name "*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

# Require a file to exist, exit with error if not found
# Usage: require_file "$CHECKPOINT"
require_file() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "Error: File not found: $file"
        exit 1
    fi
}

# Parse agent and checkpoint arguments
# Sets: CHECKPOINT, AGENT_ENTRY, AGENT_LOG, AGENT_ARGS
# Usage: parse_agent_args "$@"
#        shift $PARSED_ARGS
parse_agent_args() {
    CHECKPOINT=""
    AGENT_ENTRY=""
    AGENT_LOG="trajectory"  # Default log directory
    AGENT_ARGS=""
    PARSED_ARGS=0
    local do_continue=false

    # First pass: extract --agent, --robot, --multi_gpu
    local args=("$@")
    for ((i=0; i<${#args[@]}; i++)); do
        if [[ "${args[$i]}" == "--robot" ]]; then
            ROBOT="${args[$((i+1))]}"
            TASK=$(resolve_robot_task "$ROBOT")
        elif [[ "${args[$i]}" == "--multi_gpu" ]]; then
            parse_multi_gpu "${args[$((i+1))]}"
        elif [[ "${args[$i]}" == "--agent" ]]; then
            local alias="${args[$((i+1))]}"
            AGENT_ENTRY=$(resolve_agent_entry "$alias")
            AGENT_LOG=$(resolve_agent_log "$alias")
            if [ -z "$AGENT_ENTRY" ]; then
                echo "Error: Unknown agent alias '$alias'"
                echo "Available agents: mlp, tnet, pt"
                exit 1
            fi
            AGENT_ARGS="--agent $AGENT_ENTRY"
            echo "========================================"
            echo "Agent: $alias"
            echo "  Entry point: $AGENT_ENTRY"
            echo "  Log directory: $AGENT_LOG"
            echo "========================================"
        fi
    done

    # Second pass: parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --robot)
                PARSED_ARGS=$((PARSED_ARGS + 2))
                shift 2
                ;;
            --agent)
                PARSED_ARGS=$((PARSED_ARGS + 2))
                shift 2
                ;;
            --multi_gpu)
                PARSED_ARGS=$((PARSED_ARGS + 2))
                shift 2
                ;;
            --continue)
                do_continue=true
                PARSED_ARGS=$((PARSED_ARGS + 1))
                shift
                ;;
            --checkpoint)
                CHECKPOINT="$2"
                require_file "$CHECKPOINT"
                echo "========================================"
                echo "Using checkpoint: $CHECKPOINT"
                echo "========================================"
                PARSED_ARGS=$((PARSED_ARGS + 2))
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done

    # Handle --continue after we know the log directory
    if $do_continue; then
        CHECKPOINT=$(find_latest "$AGENT_LOG")
        if [ -z "$CHECKPOINT" ]; then
            echo "Error: No checkpoint found in $Y2R_DATA_ROOT/IsaacLab/logs/rl_games/$AGENT_LOG/"
            exit 1
        fi
        echo "========================================"
        echo "Resuming from: $CHECKPOINT"
        echo "========================================"
    fi
}

# Legacy function for backward compatibility
# Usage: parse_checkpoint_args "trajectory" "$@"
#        shift $PARSED_ARGS
parse_checkpoint_args() {
    local log_dir="$1"
    shift
    CHECKPOINT=""
    PARSED_ARGS=0

    while [[ $# -gt 0 ]]; do
        case $1 in
            --continue)
                CHECKPOINT=$(find_latest "$log_dir")
                if [ -z "$CHECKPOINT" ]; then
                    echo "Error: No checkpoint found in $Y2R_DATA_ROOT/IsaacLab/logs/rl_games/$log_dir/"
                    exit 1
                fi
                echo "========================================"
                echo "Resuming from: $CHECKPOINT"
                echo "========================================"
                PARSED_ARGS=$((PARSED_ARGS + 1))
                shift
                ;;
            --checkpoint)
                CHECKPOINT="$2"
                require_file "$CHECKPOINT"
                echo "========================================"
                echo "Using checkpoint: $CHECKPOINT"
                echo "========================================"
                PARSED_ARGS=$((PARSED_ARGS + 2))
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done
}
