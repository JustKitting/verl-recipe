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
    FSDP_WORKERS=$(python -c "import verl.workers.fsdp_workers as fw; print(fw.__file__)" 2>/dev/null || echo "")
    if [[ -n "$FSDP_WORKERS" && -f "$FSDP_WORKERS" ]]; then
        sed -i 's|cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)|cpu_offload = None if role == "actor" else (CPUOffload(offload_params=True) if getattr(fsdp_config, "param_offload", False) else None)|' "$FSDP_WORKERS" 2>/dev/null || true
        sed -i 's|cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)|cpu_offload = None if role == "actor" else (CPUOffloadPolicy(pin_memory=True) if getattr(fsdp_config, "param_offload", False) else None)|' "$FSDP_WORKERS" 2>/dev/null || true
        echo "Applied FSDP workers patch"
    fi

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
  --model MODEL          Model path (default: Qwen/Qwen2.5-3B)
  --steps N              Training steps (default: 1000)
  --batch-size N         Batch size (default: 128)
  --group-size N         Rollouts per prompt (default: 8)
  --lr RATE              Learning rate (default: 1e-5)
  --max-tokens N         Max token length (default: 2048)
  --temperature T        Sampling temperature (default: 0.7)
  --gpu-memory F         GPU memory utilization 0-1 (default: 0.15)
  --gpus N               Number of GPUs
  --wandb                Enable W&B logging
  --wandb-project NAME   W&B project name
  --atropos-config PATH  Atropos config file
  --verl-config PATH     VeRL config file
  --shell                Interactive shell

Example:
  docker run --gpus all -e WANDB_API_KEY=$WANDB_API_KEY atropos-verl \
    --model Qwen/Qwen2.5-3B --steps 1000 --wandb
EOF
}

main() {
    CLI_ARGS=()

    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h) show_help; exit 0 ;;
            --shell) detect_gpu_and_configure; apply_patches; exec /bin/bash ;;
            *) CLI_ARGS+=("$1"); shift ;;
        esac
    done

    detect_gpu_and_configure
    apply_patches

    exec python -m atropos.main "${CLI_ARGS[@]}"
}

main "$@"
