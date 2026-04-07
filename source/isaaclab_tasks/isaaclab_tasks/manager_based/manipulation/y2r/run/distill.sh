#!/bin/bash
# ==============================================================================
# Distill Y2R Teacher to Student
# ==============================================================================
# Usage:
#   ./scripts/distill.sh --t_continue                           # Latest teacher, fresh MLP student
#   ./scripts/distill.sh --student pt --t_continue              # Latest teacher, fresh PT student
#   ./scripts/distill.sh --t_agent pt --t_continue              # Latest PT teacher, fresh student
#   ./scripts/distill.sh --t_checkpoint path/to/teacher.pth     # Specific teacher, fresh student
#   ./scripts/distill.sh --t_continue --continue                # Latest teacher, resume student
#   ./scripts/distill.sh --student pt --t_continue --continue   # Resume PT student
#   ./scripts/distill.sh --t_checkpoint t.pth --checkpoint s.pth # Both specific
#   ./scripts/distill.sh --t_continue --multi_gpu 2,3           # Multi-GPU distillation
#
# Student types (--student):
#   mlp (default): DepthResNetStudent — depth encoder + flat MLP
#   pt:            DepthPointTransformerStudent — depth encoder + point transformer
# ==============================================================================

source "$(dirname "$0")/common.sh"

TEACHER_CHECKPOINT=""
STUDENT_CHECKPOINT=""
TEACHER_AGENT_ALIAS="mlp"
TEACHER_LOG_DIR="trajectory"
TEACHER_AGENT_ENTRY="rl_games_teacher_cfg_entry_point"
STUDENT_AGENT="rl_games_student_cfg_entry_point"
STUDENT_CONFIG_NAME="student_depth_distillation"
DO_TEACHER_CONTINUE=false
DO_STUDENT_CONTINUE=false
FORWARD_ARGS=()

# Parse wrapper arguments anywhere in argv and forward unknown args to Python.
while [[ $# -gt 0 ]]; do
    case $1 in
        --robot)
            ROBOT="$2"
            TASK=$(resolve_robot_task "$ROBOT")
            shift 2
            ;;
        --multi_gpu)
            parse_multi_gpu "$2"
            shift 2
            ;;
        --t_agent)
            TEACHER_AGENT_ALIAS="$2"
            TEACHER_LOG_DIR=$(resolve_agent_log "$TEACHER_AGENT_ALIAS")
            TEACHER_AGENT_ENTRY=$(resolve_agent_entry "$TEACHER_AGENT_ALIAS")
            if [ -z "$TEACHER_LOG_DIR" ] || [ -z "$TEACHER_AGENT_ENTRY" ]; then
                echo "Error: Unknown teacher agent alias '$TEACHER_AGENT_ALIAS'"
                echo "Available teacher agents: mlp, tnet, pt"
                exit 1
            fi
            echo "========================================"
            echo "Teacher agent: $TEACHER_AGENT_ALIAS"
            echo "  Entry point: $TEACHER_AGENT_ENTRY"
            echo "  Log directory: $TEACHER_LOG_DIR"
            echo "========================================"
            shift 2
            ;;
        --student)
            case "$2" in
                pt)
                    STUDENT_AGENT="rl_games_student_pt_cfg_entry_point"
                    STUDENT_CONFIG_NAME="student_depth_pt_distillation"
                    echo "========================================"
                    echo "Student: Point Transformer"
                    echo "========================================"
                    ;;
                mlp)
                    STUDENT_AGENT="rl_games_student_cfg_entry_point"
                    STUDENT_CONFIG_NAME="student_depth_distillation"
                    echo "========================================"
                    echo "Student: MLP (default)"
                    echo "========================================"
                    ;;
                *)
                    echo "Error: Unknown student type '$2'"
                    echo "Available: mlp (default), pt"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --t_continue)
            DO_TEACHER_CONTINUE=true
            shift
            ;;
        --t_checkpoint)
            TEACHER_CHECKPOINT="$2"
            require_file "$TEACHER_CHECKPOINT"
            echo "========================================"
            echo "Teacher: $TEACHER_CHECKPOINT"
            echo "========================================"
            shift 2
            ;;
        --continue)
            DO_STUDENT_CONTINUE=true
            shift
            ;;
        --checkpoint)
            STUDENT_CHECKPOINT="$2"
            require_file "$STUDENT_CHECKPOINT"
            echo "========================================"
            echo "Student: $STUDENT_CHECKPOINT"
            echo "========================================"
            shift 2
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

# Resolve student checkpoint after parsing so --student order doesn't matter.
if [ "$DO_STUDENT_CONTINUE" = true ] && [ -z "$STUDENT_CHECKPOINT" ]; then
    STUDENT_CHECKPOINT=$(find_latest "$STUDENT_CONFIG_NAME")
    if [ -z "$STUDENT_CHECKPOINT" ]; then
        echo "No student checkpoint found in $STUDENT_CONFIG_NAME, starting fresh."
    else
        echo "========================================"
        echo "Resuming student from: $STUDENT_CHECKPOINT"
        echo "========================================"
    fi
fi

# Resolve teacher checkpoint after parsing so --t_agent order doesn't matter.
if [ "$DO_TEACHER_CONTINUE" = true ] && [ -z "$TEACHER_CHECKPOINT" ]; then
    TEACHER_CHECKPOINT=$(find_latest "$TEACHER_LOG_DIR")
    if [ -z "$TEACHER_CHECKPOINT" ]; then
        echo "Error: No teacher checkpoint found in $Y2R_DATA_ROOT/IsaacLab/logs/rl_games/$TEACHER_LOG_DIR/"
        exit 1
    fi
    echo "========================================"
    echo "Teacher: $TEACHER_CHECKPOINT"
    echo "========================================"
fi

# Validate teacher checkpoint
if [ -z "$TEACHER_CHECKPOINT" ]; then
    echo "Error: Teacher checkpoint required"
    echo "Usage: ./scripts/distill.sh --t_continue"
    echo "       ./scripts/distill.sh --t_checkpoint path/to/teacher.pth"
    exit 1
fi

cd "$ISAACLAB_DIR"

if [ "$NUM_GPUS" -gt 0 ]; then
    # Multi-GPU: use torchrun with --distributed flag
    PYTHON_EXE=$(./isaaclab.sh -p -c "import sys; print(sys.executable)" 2>/dev/null | grep '^/')
    Y2R_MODE=distill Y2R_TASK=base Y2R_ROBOT=$ROBOT Y2R_GPU_OFFSET=$GPU_OFFSET \
        $PYTHON_EXE -m torch.distributed.run --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        scripts/reinforcement_learning/rl_games/distill.py \
        --task "$TASK" \
        --teacher-checkpoint "$TEACHER_CHECKPOINT" \
        --teacher-agent "$TEACHER_AGENT_ENTRY" \
        --student-agent "$STUDENT_AGENT" \
        --headless \
        --distributed \
        --enable_cameras \
        --track \
        --wandb-project-name distillation \
        --wandb-entity hgupt3 \
        ${STUDENT_CHECKPOINT:+--checkpoint "$STUDENT_CHECKPOINT"} \
        "${FORWARD_ARGS[@]}"
else
    # Single GPU
    Y2R_MODE=distill Y2R_TASK=base Y2R_ROBOT=$ROBOT ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/distill.py \
        --task "$TASK" \
        --teacher-checkpoint "$TEACHER_CHECKPOINT" \
        --teacher-agent "$TEACHER_AGENT_ENTRY" \
        --student-agent "$STUDENT_AGENT" \
        --headless \
        --enable_cameras \
        --track \
        --wandb-project-name distillation \
        --wandb-entity hgupt3 \
        ${STUDENT_CHECKPOINT:+--checkpoint "$STUDENT_CHECKPOINT"} \
        "${FORWARD_ARGS[@]}"
fi
