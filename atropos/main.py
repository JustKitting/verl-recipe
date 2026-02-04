# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Nous Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import socket

import ray
import yaml
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import auto_set_device, is_cuda_available

from .ray_trainer import RayAtroposTrainer

# Defaults (lowest priority)
DEFAULTS = {
    "model": "Qwen/Qwen2.5-3B",
    "steps": 1000,
    "batch_size": 128,
    "group_size": 8,
    "lr": 1e-5,
    "max_tokens": 2048,
    "wandb_project": "gsm8k-verl-atropos",
    "temperature": 0.7,
    "gpu_memory_utilization": 0.15,
}


def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_verl_base_config() -> OmegaConf:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    import verl.trainer

    GlobalHydra.instance().clear()
    config_path = os.path.join(os.path.dirname(verl.trainer.__file__), "config")
    initialize_config_dir(config_dir=config_path, version_base=None)
    base_config = compose(config_name="ppo_trainer")
    GlobalHydra.instance().clear()
    return base_config


def detect_gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count() or 1
    except Exception:
        return 1


def get_value(cli_val, yaml_val, default_val):
    """CLI > YAML > default"""
    if cli_val is not None:
        return cli_val
    if yaml_val is not None:
        return yaml_val
    return default_val


def build_config(cli_args, atropos_cfg: dict, verl_cfg: dict) -> OmegaConf:
    base = load_verl_base_config()
    OmegaConf.set_struct(base, False)

    env_yaml = atropos_cfg.get("env", {})
    sync_cfg = atropos_cfg.get("sync", {})
    data_cfg = verl_cfg.get("data", {})
    algorithm_cfg = verl_cfg.get("algorithm", {})
    trainer_cfg = verl_cfg.get("trainer", {})
    model_cfg = verl_cfg.get("model", {})
    actor_cfg = verl_cfg.get("actor", {})
    rollout_cfg = verl_cfg.get("rollout", {})
    ref_cfg = verl_cfg.get("ref", {})
    checkpoint_cfg = verl_cfg.get("checkpoint", {})
    ray_cfg = verl_cfg.get("ray_kwargs", {})

    # Resolve values: CLI > YAML > defaults
    model = get_value(cli_args.model, env_yaml.get("tokenizer_name"), DEFAULTS["model"])
    steps = get_value(cli_args.steps, env_yaml.get("total_steps"), DEFAULTS["steps"])
    batch_size = get_value(cli_args.batch_size, env_yaml.get("batch_size"), DEFAULTS["batch_size"])
    group_size = get_value(cli_args.group_size, env_yaml.get("group_size"), DEFAULTS["group_size"])
    lr = get_value(cli_args.lr, actor_cfg.get("lr"), DEFAULTS["lr"])
    max_tokens = get_value(cli_args.max_tokens, env_yaml.get("max_token_length"), DEFAULTS["max_tokens"])
    wandb_project = get_value(cli_args.wandb_project, env_yaml.get("wandb_name"), DEFAULTS["wandb_project"])
    temperature = get_value(cli_args.temperature, rollout_cfg.get("temperature"), DEFAULTS["temperature"])
    gpu_mem = get_value(cli_args.gpu_memory, rollout_cfg.get("gpu_memory_utilization"), DEFAULTS["gpu_memory_utilization"])

    use_wandb = cli_args.wandb if cli_args.wandb else env_yaml.get("use_wandb", False)
    gpus = get_value(cli_args.gpus, trainer_cfg.get("n_gpus_per_node"), None) or detect_gpu_count()

    # Build env dict for passing to trainer
    env = {
        "tokenizer_name": model,
        "total_steps": steps,
        "batch_size": batch_size,
        "group_size": group_size,
        "max_token_length": max_tokens,
        "use_wandb": use_wandb,
        "wandb_name": wandb_project,
        "rollout_server_url": env_yaml.get("rollout_server_url", "http://localhost:8000"),
        "steps_per_eval": env_yaml.get("steps_per_eval", 100),
        "batch_timeout": env_yaml.get("batch_timeout", 60.0),
    }

    overrides = {
        "env": env,
        "atropos": {
            "api_url": env["rollout_server_url"],
            "batch_timeout": env["batch_timeout"],
            "environment_module": atropos_cfg.get("environment_module", "atropos.environments.gsm8k"),
            "sync": sync_cfg,
        },
        "data": {
            "train_batch_size": batch_size,
            "truncation": data_cfg.get("truncation", "error"),
            "trust_remote_code": data_cfg.get("trust_remote_code", False),
        },
        "algorithm": {
            **algorithm_cfg,
            "use_kl_in_reward": False,
        },
        "trainer": {
            "project_name": wandb_project,
            "experiment_name": wandb_project,
            "total_training_steps": steps,
            "test_freq": env["steps_per_eval"],
            "logger": ["console", "wandb"] if use_wandb else ["console"],
            "default_local_dir": checkpoint_cfg.get("dir", "./checkpoints/"),
            "save_freq": checkpoint_cfg.get("save_interval", 500),
            "nnodes": trainer_cfg.get("nnodes", 1),
            "n_gpus_per_node": gpus,
            "val_before_train": trainer_cfg.get("val_before_train", False),
            "resume_mode": trainer_cfg.get("resume_mode", "disable"),
        },
        "actor_rollout_ref": {
            "hybrid_engine": True,
            "model": {
                "path": model,
                "tokenizer_path": model,
                "lora_rank": model_cfg.get("lora_rank", 0),
                "lora_alpha": model_cfg.get("lora_alpha", 32),
                "use_remove_padding": True,
                "enable_gradient_checkpointing": model_cfg.get("enable_gradient_checkpointing", True),
                "override_config": model_cfg.get("override_config", {}),
            },
            "actor": {
                "strategy": actor_cfg.get("strategy", "fsdp"),
                "optim": {"lr": float(lr)},
                "ppo_mini_batch_size": batch_size,
                "ppo_micro_batch_size_per_gpu": actor_cfg.get("ppo_micro_batch_size_per_gpu", 2),
                "ppo_epochs": actor_cfg.get("ppo_epochs", 1),
                "use_rollout_log_probs": True,
                "use_kl_loss": actor_cfg.get("use_kl_loss", True),
                "kl_loss_coef": actor_cfg.get("kl_loss_coef", 0.0),
                "kl_loss_type": actor_cfg.get("kl_loss_type", "low_var_kl"),
                "loss_agg_mode": actor_cfg.get("loss_agg_mode", "token-mean"),
                "entropy_coeff": actor_cfg.get("entropy_coeff", 0.01),
                "grad_clip": actor_cfg.get("grad_clip", 1.0),
                "clip_ratio": actor_cfg.get("clip_ratio", 0.2),
                "fsdp_config": {
                    "param_offload": actor_cfg.get("fsdp_config", {}).get("param_offload", False),
                    "optimizer_offload": actor_cfg.get("fsdp_config", {}).get("optimizer_offload", False),
                    "model_dtype": actor_cfg.get("fsdp_config", {}).get("model_dtype", "bfloat16"),
                },
            },
            "rollout": {
                "name": rollout_cfg.get("engine", "sglang"),
                "mode": rollout_cfg.get("mode", "async"),
                "n": group_size,
                "prompt_length": rollout_cfg.get("prompt_length", 1024),
                "response_length": rollout_cfg.get("response_length", 2048),
                "temperature": temperature,
                "tensor_model_parallel_size": rollout_cfg.get("tensor_model_parallel_size", 1),
                "gpu_memory_utilization": gpu_mem,
                "log_prob_micro_batch_size_per_gpu": rollout_cfg.get("log_prob_micro_batch_size_per_gpu", 8),
                "load_format": rollout_cfg.get("load_format", "safetensors"),
                "dtype": rollout_cfg.get("dtype", "bfloat16"),
                "engine_kwargs": rollout_cfg.get("engine_kwargs", {}),
                "skip_tokenizer_init": False,
                "enforce_eager": rollout_cfg.get("enforce_eager", False),
                "free_cache_engine": rollout_cfg.get("free_cache_engine", False),
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": ref_cfg.get("log_prob_micro_batch_size_per_gpu", 8),
                "fsdp_config": ref_cfg.get("fsdp_config", {}),
            },
        },
        "reward_model": {"enable": False},
        "critic": {"enable": False},
    }

    if ray_cfg:
        overrides["ray_kwargs"] = ray_cfg

    merged = OmegaConf.merge(base, OmegaConf.create(overrides))

    # Deep merge engine_kwargs (OmegaConf.merge replaces nested dicts, we need to merge them)
    engine_kwargs = rollout_cfg.get("engine_kwargs", {})
    for engine_name, engine_opts in engine_kwargs.items():
        if engine_opts:
            for key, value in engine_opts.items():
                OmegaConf.update(merged, f"actor_rollout_ref.rollout.engine_kwargs.{engine_name}.{key}", value)

    return merged


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        resolved = OmegaConf.to_container(config, resolve=True)
        pprint(resolved)
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        from verl.single_controller.ray import RayWorkerGroup

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from .fsdp_workers import AtroposActorRolloutRefWorker
            AsyncActorRolloutRefWorker = AtroposActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError(f"Strategy {config.actor_rollout_ref.actor.strategy} not supported")

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError(f"Reward model strategy {config.reward_model.strategy} not supported")
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(AsyncActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        val_reward_fn = None
        if config.get("use_val_reward_fn", False):
            val_reward_fn = load_reward_manager(
                config,
                tokenizer,
                1,
                max_resp_len=config.data.max_response_length,
                overlong_buffer_cfg=config.reward_model.overlong_buffer,
            )

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        atropos_config = OmegaConf.to_container(config.get("atropos", {}))

        trainer = RayAtroposTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=None,
            val_reward_fn=val_reward_fn,
            atropos_config=atropos_config,
        )
        trainer.init_workers()
        trainer.fit()


def run_training(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    try:
        if (
            is_cuda_available
            and config.global_profiler.tool == "nsys"
            and OmegaConf.select(config.global_profiler, "steps") is not None
            and len(OmegaConf.select(config.global_profiler, "steps")) > 0
        ):
            nsight_options = OmegaConf.to_container(
                config.global_profiler.global_tool_config.nsys.controller_nsight_options
            )
            runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
        else:
            runner = TaskRunner.remote()
        ray.get(runner.run.remote(config))
    finally:
        if ray.is_initialized():
            ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Atropos + VeRL Training")

    # Config files
    parser.add_argument("--atropos-config", default="configs/atropos.yaml")
    parser.add_argument("--verl-config", default="configs/verl.yaml")

    # Common CLI overrides
    parser.add_argument("--model", type=str, help="Model path (e.g. Qwen/Qwen2.5-3B)")
    parser.add_argument("--steps", type=int, help="Total training steps")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--group-size", type=int, help="Rollouts per prompt")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--max-tokens", type=int, help="Max token length")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--gpu-memory", type=float, help="GPU memory utilization (0-1)")
    parser.add_argument("--gpus", type=int, help="Number of GPUs")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, help="W&B project name")

    args = parser.parse_args()

    atropos_cfg = load_yaml(args.atropos_config)
    verl_cfg = load_yaml(args.verl_config)

    config = build_config(args, atropos_cfg, verl_cfg)

    auto_set_device(config)
    run_training(config)


if __name__ == "__main__":
    main()
