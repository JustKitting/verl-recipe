#!/bin/bash
# Atropos + VeRL Training Script for GSM8K
#
# Usage:
#   pip install -r atropos/requirements.txt
#   bash atropos/run_gsm8k.sh

set -e

# Configuration
MODEL=${MODEL:-"Qwen/Qwen3-0.6B"}
STEPS=${STEPS:-1000}
GPUS=${GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
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
echo "Log Dir: $LOG_DIR"
echo "Wandb: $USE_WANDB"
echo "=============================================="

if [ "$USE_WANDB" = "true" ]; then
    LOGGER='["console","wandb"]'
else
    LOGGER='["console"]'
fi

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

echo ""
echo "[2/2] Starting VeRL Trainer..."
python -m atropos.main \
    atropos.api_url="http://localhost:$ATROPOS_PORT" \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GPUS \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    data.train_batch_size=$BATCH_SIZE \
    trainer.n_gpus_per_node=$GPUS \
    trainer.total_training_steps=$STEPS \
    trainer.logger="$LOGGER" \
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
