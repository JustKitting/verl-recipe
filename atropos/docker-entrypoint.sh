#!/bin/bash
set -e

detect_gpu_and_configure() {
    command -v nvidia-smi &> /dev/null || return

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    GPU_ARCH=$(python -c "import torch; cap=torch.cuda.get_device_capability(0); print(f'{cap[0]}{cap[1]}')" 2>/dev/null || echo "0")

    echo "GPU: $GPU_NAME (x$GPU_COUNT), SM$GPU_ARCH"

    if [[ "$GPU_ARCH" -ge "120" ]]; then
        export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
        export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-LOC}
        export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
        export NCCL_NVB_DISABLE=${NCCL_NVB_DISABLE:-1}
        export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
        export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-0}
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
        export VLLM_SKIP_P2P_CHECK=${VLLM_SKIP_P2P_CHECK:-1}
    fi

    [[ "$GPU_COUNT" -gt "1" ]] && export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
}

apply_patches() {
    # Apply FSDP workers patch (respects param_offload config for ref models)
    FSDP_WORKERS=$(python -c "import verl.workers.fsdp_workers as fw; print(fw.__file__)" 2>/dev/null || echo "")
    if [[ -n "$FSDP_WORKERS" && -f "$FSDP_WORKERS" ]]; then
        sed -i 's|cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)|cpu_offload = None if role == "actor" else (CPUOffload(offload_params=True) if getattr(fsdp_config, "param_offload", False) else None)|' "$FSDP_WORKERS" 2>/dev/null || true
        sed -i 's|cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)|cpu_offload = None if role == "actor" else (CPUOffloadPolicy(pin_memory=True) if getattr(fsdp_config, "param_offload", False) else None)|' "$FSDP_WORKERS" 2>/dev/null || true
        echo "Applied FSDP workers patch"
    fi

    # Apply SGLang template_manager patch (handles dict/list chat_template for Hermes models)
    SGLANG_TM=$(python -c "import sglang.srt.managers.template_manager as tm; print(tm.__file__)" 2>/dev/null || echo "")
    if [[ -n "$SGLANG_TM" && -f "$SGLANG_TM" ]]; then
        sed -i 's/has_reasoning = re.search(force_reasoning_pattern, template) is not None/has_reasoning = re.search(force_reasoning_pattern, template) is not None if isinstance(template, str) else False/' "$SGLANG_TM" 2>/dev/null || true
        echo "Applied SGLang template_manager patch"
    fi

}

show_help() {
    cat << 'EOF'
Atropos + VeRL Training

Usage: docker run --gpus all atropos-verl [OPTIONS]

Options:
  --atropos-config PATH  Atropos config file (default: configs/atropos.yaml)
  --verl-config PATH     VeRL config file (default: configs/verl.yaml)
  --env-module MODULE    Override environment module
  --model MODEL          Override model path
  --steps N              Override training steps
  --max-tokens N         Max token length (default: 2048)
  --wandb                Enable W&B logging (requires WANDB_API_KEY env var)
  --shell                Interactive shell

Example:
  docker run --gpus all -e WANDB_API_KEY=$WANDB_API_KEY atropos-verl \
    --env-module environments.gsm8k_server --steps 1000
EOF
}

main() {
    ATROPOS_CONFIG="/workspace/atropos/configs/atropos.yaml"
    VERL_CONFIG="/workspace/atropos/configs/verl.yaml"

    # Generate unique Ray namespace to isolate this run from others
    RUN_ID=$(cat /proc/sys/kernel/random/uuid 2>/dev/null | cut -c1-8 || date +%s)
    export VERL_RAY_NAMESPACE="verl-${RUN_ID}"
    OVERRIDES="ray_kwargs.ray_init.namespace=${VERL_RAY_NAMESPACE}"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h) show_help; exit 0 ;;
            --atropos-config) ATROPOS_CONFIG="$2"; shift 2 ;;
            --verl-config) VERL_CONFIG="$2"; shift 2 ;;
            --env-module) OVERRIDES="$OVERRIDES atropos.environment_module=$2"; shift 2 ;;
            --model) OVERRIDES="$OVERRIDES model.path=$2 env.tokenizer_name=$2"; shift 2 ;;
            --steps) OVERRIDES="$OVERRIDES training.total_steps=$2 env.total_steps=$2 trainer.total_training_steps=$2"; shift 2 ;;
            --batch-size) OVERRIDES="$OVERRIDES training.batch_size=$2 env.batch_size=$2"; shift 2 ;;
            --group-size) OVERRIDES="$OVERRIDES training.group_size=$2 env.group_size=$2"; shift 2 ;;
            --max-tokens) OVERRIDES="$OVERRIDES env.max_token_length=$2"; shift 2 ;;
            --wandb) OVERRIDES="$OVERRIDES env.use_wandb=true"; shift ;;
            --shell) detect_gpu_and_configure; apply_patches; exec /bin/bash ;;
            *) OVERRIDES="$OVERRIDES $1"; shift ;;
        esac
    done

    detect_gpu_and_configure
    apply_patches

    echo "Atropos config: $ATROPOS_CONFIG"
    echo "VeRL config: $VERL_CONFIG"
    [[ -n "$OVERRIDES" ]] && echo "Overrides:$OVERRIDES"

    exec python -m atropos.main \
        --atropos-config "$ATROPOS_CONFIG" \
        --verl-config "$VERL_CONFIG" \
        $OVERRIDES
}

main "$@"
