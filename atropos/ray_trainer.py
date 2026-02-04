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

import logging
import time
from collections import defaultdict

import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss

from .utils.debug import debug_batch_data, save_batch_tensors
from .utils.env_adapter import VeRLScoreAdapter, compute_advantage_with_score_adapter
from .utils.sync_coordinator import create_coordinator

from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
import numpy as np

from .data_source import AtroposDataSource


def safe_reduce_metrics(metrics: dict) -> dict:
    result = {}
    for key, val in metrics.items():
        try:
            scalar_vals = [v for v in val if np.isscalar(v) or (hasattr(v, 'shape') and v.shape == ())]
            if not scalar_vals:
                continue
            if "max" in key:
                result[key] = np.max(scalar_vals)
            elif "min" in key:
                result[key] = np.min(scalar_vals)
            else:
                result[key] = np.mean(scalar_vals)
        except (ValueError, TypeError):
            continue
    return result
from .utils.http import log_section

logger = logging.getLogger(__name__)


class RayAtroposTrainer(RayPPOTrainer):

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        atropos_config: dict = None,
        **kwargs,
    ):
        self._atropos_config = atropos_config or {}

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            **kwargs,
        )

        env_cfg = config.get("env", {})

        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have pad_token_id set.")

        self.atropos_source = AtroposDataSource(
            api_url=self._atropos_config.get("api_url", "http://localhost:8000"),
            pad_token_id=tokenizer.pad_token_id,
            max_seq_len=env_cfg.max_token_length,
            truncation_policy=config.data.get("truncation", "error"),
        )
        self.atropos_batch_timeout = self._atropos_config.get("batch_timeout", 60.0)
        self.atropos_register_kwargs = self._atropos_config.get("register_kwargs", {})
        self._inference_urls = []
        self._env_process = None
        self._env_log_file = None

        sync_config = self._atropos_config.get("sync", {})
        self.sync_queue_threshold = sync_config.get("queue_threshold", 1)
        self.sync_max_steps = sync_config.get("max_steps_between_sync", 4)
        self.sync_min_steps = sync_config.get("min_steps_between_sync", 1)
        self.sync_log_drift = sync_config.get("log_drift", True)

        # Coordinator configuration
        coordinator_config = sync_config.get("coordinator", {})
        self._coordinator_request_timeout = coordinator_config.get("request_timeout", 300.0)
        self._coordinator_cleanup_interval = coordinator_config.get("cleanup_interval", 30.0)

        self._steps_since_sync = 0
        self._last_sync_step = 0
        self._total_syncs = 0

        debug_config = self._atropos_config.get("debug", {})
        self._debug_enabled = debug_config.get("enabled", False)
        self._debug_output_dir = debug_config.get("output_dir", "./logs")
        self._debug_save_tensors_at_steps = set(debug_config.get("save_tensors_at_steps", []))

        self._sync_coordinator = create_coordinator(
            request_timeout=self._coordinator_request_timeout,
            cleanup_interval=self._coordinator_cleanup_interval,
        )
        logger.info(
            f"Created sync coordinator: request_timeout={self._coordinator_request_timeout}s, "
            f"cleanup_interval={self._coordinator_cleanup_interval}s"
        )

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        self.train_dataset = None
        self.val_dataset = val_dataset
        self.train_dataloader = None
        self.val_dataloader = None

        if self.config.trainer.total_training_steps is None:
            raise ValueError("trainer.total_training_steps must be set for Atropos integration.")
        self.total_training_steps = self.config.trainer.total_training_steps
        logger.info(f"Total training steps: {self.total_training_steps}")

    def _save_checkpoint(self):
        class DummyDataloader:
            def state_dict(self):
                return {"atropos": True, "step": 0}

        original_dataloader = self.train_dataloader
        self.train_dataloader = DummyDataloader()
        try:
            super()._save_checkpoint()
        finally:
            self.train_dataloader = original_dataloader

    def _discover_inference_urls(self) -> list[str]:
        if hasattr(self, 'async_rollout_manager') and self.async_rollout_manager is not None:
            server_addrs = getattr(self.async_rollout_manager, 'server_addresses', [])
            if server_addrs:
                urls = [f"http://{addr}/v1" for addr in server_addrs]
                logger.info(f"Discovered {len(urls)} inference URL(s): {urls}")
                return urls

        raise RuntimeError("Could not discover inference URLs from async_rollout_manager.")

    def _register_with_atropos(self):
        env_cfg = self.config.get("env", {})

        register_kwargs = {
            "wandb_group": self.config.trainer.experiment_name,
            "wandb_project": self.config.trainer.project_name,
            **self.atropos_register_kwargs,
        }

        self.atropos_source.register(
            batch_size=env_cfg.batch_size,
            max_token_len=env_cfg.max_token_length,
            num_steps=env_cfg.total_steps,
            starting_step=self.global_steps,
            **register_kwargs,
        )

    def _start_trajectory_api(self):
        import os
        import subprocess
        import sys
        from urllib.parse import urlparse

        api_url = self._atropos_config.get("api_url", "http://localhost:8000")
        port = str(urlparse(api_url).port or 8000)

        log_dir = self._debug_output_dir or "./logs"
        os.makedirs(log_dir, exist_ok=True)
        api_log_path = os.path.join(log_dir, "trajectory_api.log")

        self._api_log_file = open(api_log_path, "w")
        self._api_process = subprocess.Popen(
            [sys.executable, "-m", "atroposlib.cli.run_api", "--port", port],
            stdout=self._api_log_file,
            stderr=subprocess.STDOUT,
        )
        logger.info(f"Trajectory API started (PID: {self._api_process.pid}, port: {port})")

        import requests
        for i in range(30):
            try:
                requests.get(f"{api_url}/health", timeout=1)
                logger.info(f"Trajectory API ready at {api_url}")
                return
            except Exception:
                if i % 5 == 0:
                    logger.info(f"Waiting for trajectory API...")
                time.sleep(1)
        raise RuntimeError(f"Trajectory API did not start in 30s")

    def _start_environment(self):
        import os
        import subprocess
        import sys

        env_cfg = self.config.get("env", {})

        env_module = self._atropos_config.get("environment_module")
        if not env_module:
            raise ValueError("atropos.environment_module is required but not set in config.")

        logger.info(f"Starting environment: module={env_module}, batch_size={env_cfg.batch_size}, group_size={env_cfg.group_size}")

        proc_env = os.environ.copy()

        ray_address = "auto"
        ray_namespace = "verl"
        if ray.is_initialized():
            ray_address = ray.get_runtime_context().gcs_address
            ray_namespace = ray.get_runtime_context().namespace or "verl"
            logger.info(f"Environment will connect to Ray: address={ray_address}, namespace={ray_namespace}")

        api_url = self._atropos_config.get("api_url", "http://localhost:8000")
        cmd = [
            sys.executable, "-m", "atropos.environments.verl_adapter", "serve",
            "--env-module", env_module,
            "--tokenizer", str(env_cfg.tokenizer_name),
            "--ray-namespace", ray_namespace,
            "--ray-address", ray_address,
            "--env.tokenizer_name", str(env_cfg.tokenizer_name),
            "--env.group_size", str(env_cfg.group_size),
            "--env.batch_size", str(env_cfg.batch_size),
            "--env.total_steps", str(env_cfg.total_steps),
            "--env.steps_per_eval", str(env_cfg.steps_per_eval),
            "--env.max_token_length", str(env_cfg.max_token_length),
            "--env.use_wandb", str(env_cfg.use_wandb).lower(),
            "--env.wandb_name", str(env_cfg.wandb_name),
            "--env.rollout_server_url", api_url,
        ]

        log_dir = self._debug_output_dir or "./logs"
        env_log_path = os.path.join(log_dir, "gsm8k_env.log")
        os.makedirs(log_dir, exist_ok=True)

        self._env_log_file = open(env_log_path, "w")
        self._env_process = subprocess.Popen(
            cmd,
            env=proc_env,
            stdout=self._env_log_file,
            stderr=subprocess.STDOUT,
        )
        logger.info(f"Environment started (PID: {self._env_process.pid}, log: {env_log_path})")

    def _cleanup_trajectory_api(self):
        import subprocess
        if hasattr(self, '_api_process') and self._api_process is not None:
            if self._api_process.poll() is None:
                self._api_process.terminate()
                try:
                    self._api_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._api_process.kill()
            self._api_process = None
        if hasattr(self, '_api_log_file') and self._api_log_file is not None:
            self._api_log_file.close()
            self._api_log_file = None

    def _cleanup_environment(self):
        import subprocess
        if hasattr(self, '_env_process') and self._env_process is not None:
            if self._env_process.poll() is None:
                logger.info(f"Terminating environment subprocess (PID: {self._env_process.pid})...")
                self._env_process.terminate()
                try:
                    self._env_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Environment subprocess did not terminate, killing...")
                    self._env_process.kill()
            self._env_process = None

        if hasattr(self, '_env_log_file') and self._env_log_file is not None:
            self._env_log_file.close()
            self._env_log_file = None

    def _should_sync_weights(self) -> tuple[bool, dict]:
        metrics = {}
        status = self.atropos_source.get_status()
        queue_size = status.get("queue_size", 0)
        current_step = status.get("current_step", 0)

        metrics["staleness/queue_size"] = queue_size
        metrics["staleness/api_step"] = current_step
        metrics["staleness/steps_since_sync"] = self._steps_since_sync
        metrics["staleness/total_syncs"] = self._total_syncs

        if self._steps_since_sync < self.sync_min_steps:
            return False, metrics

        # Sync when queue is low or max steps reached
        if queue_size <= self.sync_queue_threshold:
            metrics["staleness/sync_reason"] = "queue_low"
            return True, metrics
        elif self._steps_since_sync >= self.sync_max_steps:
            metrics["staleness/sync_reason"] = "max_steps"
            return True, metrics

        return False, metrics

    def _compute_drift_metrics(self, batch: DataProto, computed_log_probs: torch.Tensor) -> dict:
        metrics = {}

        if "rollout_log_probs" not in batch.batch.keys():
            return metrics

        rollout_lp = batch.batch["rollout_log_probs"]
        current_lp = computed_log_probs
        response_mask = batch.batch["response_mask"]

        lp_diff = (current_lp - rollout_lp).abs()

        if response_mask.sum() > 0:
            drift = (lp_diff * response_mask).sum() / response_mask.sum()
            metrics["drift/avg_logprob_diff"] = drift.item()

            log_ratio = current_lp - rollout_lp
            ratio = torch.exp(log_ratio.clamp(-10, 10))
            avg_ratio = (ratio * response_mask).sum() / response_mask.sum()
            metrics["drift/avg_importance_ratio"] = avg_ratio.item()

            max_diff = (lp_diff * response_mask).max()
            metrics["drift/max_logprob_diff"] = max_diff.item()

        return metrics

    def _update_sync_state(self):
        self._steps_since_sync = 0
        self._last_sync_step = self.global_steps
        self._total_syncs += 1

    def _load_checkpoint(self) -> bool:
        import os

        from verl.utils.checkpoint.checkpoint_handler import find_latest_ckpt_path

        if self.config.trainer.resume_mode == "disable":
            return False

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return False
        elif self.config.trainer.resume_mode == "resume_path":
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)

        print(f"Load from checkpoint folder: {global_step_folder}")

        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Resuming from step {self.global_steps}")

        actor_path = os.path.join(global_step_folder, "actor")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )

        return True

    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        tracker = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self._inference_urls = self._discover_inference_urls()
        ray.get(self._sync_coordinator.set_inference_urls.remote(self._inference_urls))
        self._start_trajectory_api()
        self._start_environment()
        self._register_with_atropos()

        self._force_initial_sync = True

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Atropos Training",
        )

        self.global_steps += 1
        timing_raw = defaultdict(float)

        try:
            while self.global_steps <= self.total_training_steps:
                metrics = {}

                with marked_timer("fetch_batch", timing_raw, color="cyan"):
                    batch = self.atropos_source.get_batch(timeout=self.atropos_batch_timeout)
                    if batch is None:
                        time.sleep(0.5)
                        continue

                if self._debug_enabled:
                    debug_batch_data(batch, self.global_steps, "after_fetch", self._debug_output_dir)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if "token_level_scores" not in batch.batch.keys():
                        raise ValueError("Atropos batch missing 'token_level_scores'")
                    batch = VeRLScoreAdapter.scores_to_rewards(batch)
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        if "rollout_log_probs" not in batch.batch.keys():
                            raise ValueError("Atropos batch missing 'rollout_log_probs'.")
                        computed_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = computed_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.get("loss_scale_factor", 1.0),
                        )
                        metrics["actor/entropy"] = entropy_agg.detach().item()
                        batch.batch["old_log_probs"] = computed_log_prob.batch["old_log_probs"]

                    # Log drift for monitoring
                    if "rollout_log_probs" in batch.batch:
                        stale_rollout_lp = batch.batch["rollout_log_probs"]
                        fresh_old_lp = batch.batch["old_log_probs"]
                        response_mask = batch.batch["response_mask"]

                        if response_mask.sum() > 0:
                            diff = (fresh_old_lp - stale_rollout_lp).abs()
                            mean_diff = (diff * response_mask).sum() / response_mask.sum()
                            max_diff = (diff * response_mask).max()
                            metrics["staleness/rollout_lp_drift"] = mean_diff.item()
                            metrics["staleness/rollout_lp_max_drift"] = max_diff.item()

                    # Apply rollout correction (IS weights + rejection sampling)
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
                        batch, rollout_corr_metrics = compute_rollout_correction_and_add_to_batch(
                            batch, rollout_corr_config
                        )
                        metrics.update(rollout_corr_metrics)

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                            ref_log_prob = None  # Free after union

                    with marked_timer("adv", timing_raw, color="green"):
                        batch = compute_advantage_with_score_adapter(
                            data=batch,
                            compute_advantage_fn=compute_advantage,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self._debug_enabled:
                        debug_batch_data(batch, self.global_steps, "after_advantage", self._debug_output_dir)

                    if self.sync_log_drift:
                        drift_metrics = self._compute_drift_metrics(
                            batch, computed_log_prob.batch["old_log_probs"]
                        )
                        metrics.update(drift_metrics)

                    # Free computed_log_prob to release GPU memory
                    computed_log_prob = None

                    self._steps_since_sync += 1
                    should_sync, staleness_metrics = self._should_sync_weights()
                    metrics.update(staleness_metrics)

                    # Handle initial sync (force sync on first step after checkpoint)
                    # Don't clear the flag here - only clear it after sync actually happens
                    if self._force_initial_sync:
                        should_sync = True
                        logger.info("Forcing initial weight sync...")

                    batch.meta_info["do_sync"] = should_sync

                    if self._debug_enabled:
                        debug_batch_data(batch, self.global_steps, "before_update", self._debug_output_dir)
                        if self.global_steps in self._debug_save_tensors_at_steps:
                            save_batch_tensors(batch, self.global_steps, self._debug_output_dir)

                    # Check response_mask before actor update - skip if too many samples empty
                    rm = batch.batch["response_mask"]
                    rm_per_sample = rm.sum(dim=-1)
                    valid_samples = (rm_per_sample > 0).sum().item()
                    total_samples = rm.shape[0]
                    total_valid_tokens = rm.sum().item()

                    # CRITICAL: Skip if NO valid response tokens at all - would crash in verl
                    if valid_samples == 0 or total_valid_tokens == 0:
                        logger.error(f"[SKIP] CRITICAL: Batch has 0 valid response tokens - skipping to avoid crash")
                        # Still sync if needed to prevent staleness from compounding
                        if should_sync:
                            logger.info(f"[SYNC] Syncing weights despite skip (staleness prevention)")
                            self.actor_rollout_wg.sync_weights_only()
                            self._update_sync_state()
                        self.global_steps += 1
                        progress_bar.update(1)
                        continue

                    # Skip batch if less than half samples are valid (too much filtering causes chunk issues)
                    # BUT don't skip if we need to do initial sync (checkpoint weights -> rollout server)
                    if valid_samples < total_samples // 2 and not self._force_initial_sync:
                        logger.warning(f"[SKIP] Only {valid_samples}/{total_samples} valid samples - skipping batch")
                        # Still sync if needed to prevent staleness from compounding
                        if should_sync:
                            logger.info(f"[SYNC] Syncing weights despite skip (staleness prevention)")
                            self.actor_rollout_wg.sync_weights_only()
                            self._update_sync_state()
                        self.global_steps += 1
                        progress_bar.update(1)
                        continue
                    elif valid_samples < total_samples // 2 and self._force_initial_sync:
                        logger.warning(f"[SYNC] {valid_samples}/{total_samples} valid but forcing initial sync to update rollout server weights")

                    with marked_timer("update_actor", timing_raw, color="magenta"):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = safe_reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if should_sync:
                        self._update_sync_state()
                        # Clear initial sync flag only after sync actually happens
                        if self._force_initial_sync:
                            self._force_initial_sync = False
                            logger.info("Initial weight sync completed")

                data_metrics = compute_data_metrics(batch=batch, use_critic=False)
                metrics.update(data_metrics)

                timing_metrics = compute_timing_metrics(batch, timing_raw)
                metrics.update(timing_metrics)

                tracker.log(data=metrics, step=self.global_steps)

                if self.global_steps % self.config.trainer.save_freq == 0 or is_last_step:
                    self._save_checkpoint()

                if self.val_reward_fn is not None and self.global_steps % self.config.trainer.test_freq == 0:
                    val_metrics = self._validate()
                    if val_metrics:
                        tracker.log(data=val_metrics, step=self.global_steps)

                progress_bar.update(1)
                progress_bar.set_postfix({"loss": metrics.get("actor/loss", 0)})
                self.global_steps += 1
                timing_raw.clear()

                # Free batch and actor_output to release memory before next iteration
                batch = None
                actor_output = None

                # Clear GPU cache to prevent memory fragmentation
                torch.cuda.empty_cache()

            log_section(f"Training complete. Total syncs: {self._total_syncs}")
        finally:
            self._cleanup_environment()
            self._cleanup_trajectory_api()
            progress_bar.close()
