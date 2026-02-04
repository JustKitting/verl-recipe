#!/bin/bash
# Atropos + VeRL Training Script for GSM8K
#
# Usage:
#   pip install -r atropos/requirements.txt
#   bash atropos/run_gsm8k.sh

set -e

# Configuration
MODEL=${MODEL:-"Qwen/Qwen2.5-3B"}
STEPS=${STEPS:-1000}
GPUS=${GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-128}
GROUP_SIZE=${GROUP_SIZE:-8}
ATROPOS_PORT=${ATROPOS_PORT:-8000}
LOG_DIR=${LOG_DIR:-"./logs/gsm8k"}
USE_WANDB=${USE_WANDB:-false}

echo "=============================================="
echo "Atropos + VeRL GSM8K Training"
echo "=============================================="
echo "Model: $MODEL"
echo "Steps: $STEPS"
echo "GPUs: $GPUS"
echo "Batch Size: $BATCH_SIZE"
echo "Group Size: $GROUP_SIZE"
echo "Log Dir: $LOG_DIR"
echo "Wandb: $USE_WANDB"
echo "=============================================="

mkdir -p "$LOG_DIR"

cleanup() {
    echo "Cleaning up..."
    pkill -f "atroposlib.cli.run_api" 2>/dev/null || true
    pkill -f "atropos.main" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
}
trap cleanup EXIT

echo ""
echo "[1/2] Starting Atropos Trajectory API..."
python -m atroposlib.cli.run_api --port $ATROPOS_PORT > "$LOG_DIR/atropos_api.log" 2>&1 &
ATROPOS_PID=$!
echo "  PID: $ATROPOS_PID"
echo "  Log: $LOG_DIR/atropos_api.log"

echo "  Waiting for API..."
for i in {1..30}; do
    if curl -s "http://localhost:$ATROPOS_PORT/" > /dev/null 2>&1; then
        echo "  API ready!"
        break
    fi
    sleep 1
done

# Build overrides
OVERRIDES=""
OVERRIDES="$OVERRIDES env.tokenizer_name=$MODEL"
OVERRIDES="$OVERRIDES env.batch_size=$BATCH_SIZE"
OVERRIDES="$OVERRIDES env.group_size=$GROUP_SIZE"
OVERRIDES="$OVERRIDES env.total_steps=$STEPS"
OVERRIDES="$OVERRIDES env.use_wandb=$USE_WANDB"
OVERRIDES="$OVERRIDES rollout.tensor_model_parallel_size=$GPUS"
OVERRIDES="$OVERRIDES trainer.n_gpus_per_node=$GPUS"

echo ""
echo "[2/2] Starting VeRL Trainer..."
python -m atropos.main \
    --atropos-config configs/atropos.yaml \
    --verl-config configs/verl.yaml \
    $OVERRIDES \
    > "$LOG_DIR/verl_trainer.log" 2>&1 &
TRAINER_PID=$!
echo "  PID: $TRAINER_PID"
echo "  Log: $LOG_DIR/verl_trainer.log"

echo ""
echo "Training started! Monitor: tail -f $LOG_DIR/verl_trainer.log"
echo "Press Ctrl+C to stop."
echo ""

wait $TRAINER_PID
echo "Training complete!"
