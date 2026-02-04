# Recipe: Atropos Integration

> Integration with [Atropos](https://github.com/NousResearch/atropos), the RL environment framework from Nous Research.

## Architecture

VeRL trains models using data from Atropos environments. The integration uses a Ray-based coordination pattern where a named actor shares state between the trainer and environment processes.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                    VeRL                                     │
│  ┌─────────────────────┐    ┌──────────────────┐    ┌───────────────────┐   │
│  │  SGLang Inference   │    │  SyncCoordinator │    │  RayAtroposTrainer│   │
│  │  (async rollout)    │    │  (Ray Actor)     │    │   - GRPO updates  │   │
│  │  - /v1/completions  │    │  - inference URLs│    │   - weight sync   │   │
│  └─────────┬───────────┘    │  - sync gating   │    └─────────┬─────────┘   │
│            │                └────────▲─────────┘              │             │
│            │                         │ query URLs             │ GET /batch  │
└────────────┼─────────────────────────┼────────────────────────┼─────────────┘
             │                         │                        │
             │ /v1/chat/completions    │                        │
             ▼                         │                        ▼
┌────────────────────────────┐         │          ┌───────────────────────┐
│   Atropos Environment      │─────────┘          │   Trajectory API      │
│   (wrapped by VeRLAdapter) │                    │   (localhost:8000)    │
│   - generates rollouts     │───────────────────▶│   - queues batches    │
│   - scores responses       │   POST /scored     │   - tracks steps      │
└────────────────────────────┘                    └───────────────────────┘
```

### Key Components

| Component         | Role                                                   |
|-------------------|--------------------------------------------------------|
| `main.py`         | Entry point, builds config, launches Ray trainer       |
| `ray_trainer.py`  | GRPO training loop, spawns environment subprocess      |
| `SyncCoordinator` | Named Ray actor storing inference URLs and sync state  |
| `verl_adapter.py` | CLI that wraps any Atropos environment for VeRL        |
| `env_adapter.py`  | Adapter class that overrides server configs            |
| `fsdp_workers.py` | Extended workers with weight sync and request draining |
| `data_source.py`  | Client for Trajectory API                              |

### Inference URL Discovery

The trainer discovers SGLang server URLs from the async rollout manager and stores them on a named Ray actor (`SyncCoordinator`). The environment subprocess connects to Ray and queries this actor for the URLs - no environment variables needed.

```python
# Trainer stores URLs
coordinator.set_inference_urls.remote(["http://192.168.1.1:30000/v1", ...])

# Environment queries URLs
urls = ray.get(coordinator.get_inference_urls.remote())
```

## Docker (Recommended)

```bash
# Build
docker build -t atropos-verl .

# Run training (creates new container)
docker run --gpus all --shm-size=32g --pid=host \
  --name atropos-training \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  atropos-verl --steps 1000

# With W&B logging
docker run --gpus all --shm-size=32g --pid=host \
  --name atropos-training \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  atropos-verl --steps 1000 --wandb

# Interactive shell
docker run --gpus all -it atropos-verl --shell
```

### Managing an Existing Container

If you've created a container with `--name atropos-training`, use these commands:

```bash
# Check container status
docker ps -a --filter "name=atropos-training" --format "table {{.Status}}\t{{.Names}}"

# Start the container (runs training automatically via CMD)
docker start atropos-training

# Stop training (kills all processes, container stays)
docker exec atropos-training pkill -9 -f "python -m atropos"
docker exec atropos-training pkill -9 -f ray

# Clean up Ray state (do this before restarting)
docker exec atropos-training rm -rf /tmp/ray/*

# Clear Python cache (after code changes)
docker exec atropos-training find /workspace -name "*.pyc" -delete
docker exec atropos-training find /workspace -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Restart training from scratch (RECOMMENDED)
docker exec atropos-training pkill -9 -f "python -m atropos" 2>/dev/null
docker exec atropos-training pkill -9 -f ray 2>/dev/null
sleep 3
docker exec atropos-training rm -rf /tmp/ray/*
docker exec -d atropos-training /docker-entrypoint.sh --wandb

# Check training status
docker exec atropos-training curl -s http://localhost:8000/status

# View logs
docker logs --tail 50 atropos-training
docker exec atropos-training tail -50 /workspace/training.log

# Check GPU usage
docker exec atropos-training nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

**Important**: Never run the entrypoint multiple times without killing existing processes first. Each run creates a new training process with a unique Ray namespace, causing conflicts.

### Docker Options

| Option           | Description          |
|------------------|----------------------|
| `--steps N`      | Training steps       |
| `--model PATH`   | Model/tokenizer path |
| `--batch-size N` | Batch size           |
| `--group-size N` | Rollouts per prompt  |
| `--max-tokens N` | Max token length     |
| `--wandb`        | Enable W&B logging   |
| `--shell`        | Interactive shell    |

## Adding New Environments

Environments are exact copies of upstream Atropos files. The adapter handles all VeRL integration.

### Steps

1. Copy environment from upstream Atropos:
   ```bash
   cp /path/to/atropos/environments/letter_counting.py environments/
   ```

2. Update config (`configs/atropos.yaml`):
   ```yaml
   environment_module: "atropos.environments.letter_counting"
   ```

3. Done. The adapter automatically:
   - Wraps the environment class
   - Injects inference server URLs from the coordinator
   - Overrides config with CLI args from `atropos.yaml`

### How It Works

```
verl_adapter.py (CLI)
    │
    ├─► load_env("atropos.environments.gsm8k")    # imports your file
    │
    └─► create_verl_adapter(GSM8kEnv)             # wraps it
            │
            └─► VeRLAdapter.config_init()
                    │
                    ├─► super().config_init()     # gets base config
                    ├─► get_verl_server_configs() # queries coordinator
                    └─► returns (config, servers)
```

The environment file generally can be an **exact copy** of upstream - no modifications needed.

## Configuration

### `configs/atropos.yaml` - Environment Settings

```yaml
environment_module: "atropos.environments.gsm8k"

env:
  group_size: 8                     # rollouts per prompt
  batch_size: 128                   # prompts per batch
  tokenizer_name: "Qwen/Qwen2.5-3B"
  total_steps: 5000
  max_token_length: 2048
  use_wandb: false
```

### `configs/verl.yaml` - Training Settings

See `configs/verl.yaml` for full options. Key settings:

```yaml
trainer:
  resume_mode: disable  # Checkpoint resume not supported

actor:
  use_kl_loss: true
  kl_loss_coef: 0.0     # KL tracked but not penalized
```

## Data Format

| Atropos Field        | VeRL Field           | Description                            |
|----------------------|----------------------|----------------------------------------|
| `tokens`             | `input_ids`          | Full token sequence                    |
| `masks`              | `response_mask`      | -100 for prompt, token_id for response |
| `scores`             | `token_level_scores` | Reward (on last response token)        |
| `inference_logprobs` | `rollout_log_probs`  | Logprobs from generation               |
| `advantages`         | `atropos_advantages` | Optional per-token advantages          |

## File Structure

```
atropos/
├── main.py                 # Entry point
├── ray_trainer.py          # GRPO trainer with Atropos integration
├── data_source.py          # Trajectory API client
├── fsdp_workers.py         # Weight sync with request draining
├── configs/
│   ├── atropos.yaml        # Environment config
│   └── verl.yaml           # Training config
├── environments/
│   ├── gsm8k.py            # Upstream GSM8K (exact copy)
│   └── verl_adapter.py     # CLI wrapper
├── utils/
│   ├── sync_coordinator.py # Ray actor for state sharing
│   ├── env_adapter.py      # VeRL adapter class
│   └── patches.py          # Runtime patches
├── Dockerfile
└── docker-entrypoint.sh
```

## Known Issues & Patches

### Checkpoint Resume Not Supported
Checkpoint resumption is not supported. Training must complete in a single run. The `trainer.resume_mode` is set to `disable` by default. If training is interrupted, restart from the beginning with a fresh model.

### SGLang Dict Chat Template
Applied automatically by `docker-entrypoint.sh`:
Models with tool-calling (Hermes) have dict chat_template. SGLang crashes with `TypeError`.

### FSDP CPU Offload
VeRL hardcodes CPU offload for ref models. Patch respects `fsdp_config.param_offload`.

## References

- [Atropos Repository](https://github.com/NousResearch/atropos)
- [VeRL Documentation](https://github.com/volcengine/verl)
