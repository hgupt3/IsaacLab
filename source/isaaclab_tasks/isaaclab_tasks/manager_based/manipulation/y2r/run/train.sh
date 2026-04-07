#!/bin/bash
# ==============================================================================
# Train Y2R Teacher Policy
# ==============================================================================
# Usage:
#   ./y2r_sim/run/train.sh                           # Fresh training (MLP)
#   ./y2r_sim/run/train.sh --continue                # Resume MLP from latest
#   ./y2r_sim/run/train.sh --agent tnet              # Train PointNet+TNet
#   ./y2r_sim/run/train.sh --agent tnet --continue   # Resume TNet from latest
#   ./y2r_sim/run/train.sh --agent pt                # Train Point Transformer
#   ./y2r_sim/run/train.sh --checkpoint path/to/model.pth
#   ./y2r_sim/run/train.sh --multi_gpu 2,3           # Train on GPUs 2 and 3
#   ./y2r_sim/run/train.sh --multi_gpu 0,1,2,3 --continue  # 4-GPU resume
#
# Available agents: mlp (default), tnet, pt
# ==============================================================================

source "$(dirname "$0")/common.sh"

parse_agent_args "$@"
shift $PARSED_ARGS

cd "$ISAACLAB_DIR"

if [ "$NUM_GPUS" -gt 0 ]; then
    # Multi-GPU: use torchrun with --distributed flag
    # GPU_OFFSET shifts LOCAL_RANK so e.g. --multi_gpu 2,3 uses physical GPUs 2,3
    # We use a wrapper script to apply the offset before the training script starts
    PYTHON_EXE=$(./isaaclab.sh -p -c "import sys; print(sys.executable)" 2>/dev/null | grep '^/')
    Y2R_MODE=train Y2R_TASK=base Y2R_ROBOT=$ROBOT Y2R_GPU_OFFSET=$GPU_OFFSET \
        $PYTHON_EXE -m torch.distributed.run --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        scripts/reinforcement_learning/rl_games/train.py \
        --task "$TASK" \
        --headless \
        --distributed \
        --track \
        --wandb-project-name trajectory \
        --wandb-entity hgupt3 \
        ${AGENT_ARGS} \
        ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
        "$@"
else
    # Single GPU
    Y2R_MODE=train Y2R_TASK=base Y2R_ROBOT=$ROBOT ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
        --task "$TASK" \
        --headless \
        --track \
        --wandb-project-name trajectory \
        --wandb-entity hgupt3 \
        ${AGENT_ARGS} \
        ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
        "$@"
fi
