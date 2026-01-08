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

from .data_source import AtroposDataSource
from .utils.http import log_section

logger = logging.getLogger(__name__)


def compute_advantage_with_atropos_override(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> DataProto:
    data = compute_advantage(
        data=data,
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )

    if "atropos_advantages" in data.batch.keys():
        data.batch["advantages"] = data.batch["advantages"] + data.batch["atropos_advantages"]

    return data


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

        self._steps_since_sync = 0
        self._last_sync_step = 0
        self._total_syncs = 0

        debug_config = self._atropos_config.get("debug", {})
        self._debug_enabled = debug_config.get("enabled", False)
        self._debug_output_dir = debug_config.get("output_dir", "./logs")
        self._debug_save_tensors_at_steps = set(debug_config.get("save_tensors_at_steps", []))

        self._sync_coordinator = create_coordinator()
        logger.info("Created sync coordinator for request gating")

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

        api_url = self._atropos_config.get("api_url", "http://localhost:8000")
        cmd = [
            sys.executable, "-m", "atropos.environments.verl_adapter", "serve",
            "--env-module", env_module,
            "--tokenizer", str(env_cfg.tokenizer_name),
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

        if queue_size <= self.sync_queue_threshold:
            metrics["staleness/sync_reason"] = "queue_low"
            return True, metrics

        if self._steps_since_sync >= self.sync_max_steps:
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

    def _load_checkpoint(self):
        import os

        from verl.utils.checkpoint.checkpoint_handler import find_latest_ckpt_path

        if self.config.trainer.resume_mode == "disable":
            return 0

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
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

        print("Skipping dataloader state restoration")

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
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
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

                        # DEBUG: Compare rollout vs actor log probs
                        try:
                            rlp = batch.batch["rollout_log_probs"]
                            alp = computed_log_prob.batch["old_log_probs"]
                            rmask = batch.batch["response_mask"]
                            ids = batch.batch["input_ids"]
                            print(f"[LOGPROB DEBUG] Step {self.global_steps}: rlp shape={rlp.shape}, alp shape={alp.shape}, rmask shape={rmask.shape}", flush=True)
                            ex = 0
                            resp_len = int(rmask[ex].sum().item())
                            r_lp = rlp[ex, :resp_len]
                            a_lp = alp[ex, :resp_len]
                            diff = (a_lp - r_lp).abs()
                            print(f"[LOGPROB DEBUG] resp_len={resp_len}", flush=True)
                            print(f"[LOGPROB DEBUG] Rollout first 10: {r_lp[:10].tolist()}", flush=True)
                            print(f"[LOGPROB DEBUG] Actor   first 10: {a_lp[:10].tolist()}", flush=True)
                            print(f"[LOGPROB DEBUG] Diff    first 10: {diff[:10].tolist()}", flush=True)
                            print(f"[LOGPROB DEBUG] Mean diff={diff.mean().item():.4f}, Max={diff.max().item():.4f}", flush=True)
                        except Exception as e:
                            print(f"[LOGPROB DEBUG] ERROR: {e}", flush=True)

                        batch.batch["old_log_probs"] = computed_log_prob.batch["old_log_probs"]

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                            ref_log_prob = None  # Free after union

                    with marked_timer("adv", timing_raw, color="green"):
                        batch = compute_advantage_with_atropos_override(
                            data=batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # DEBUG: Check if advantages are correctly signed
                    if self.global_steps <= 3:
                        scores = batch.batch["token_level_scores"]
                        advantages = batch.batch["advantages"]
                        response_mask = batch.batch["response_mask"]
                        input_ids = batch.batch["input_ids"]
                        # Get per-sequence scores (sum of token-level, which is just the single score on last token)
                        seq_scores = (scores * response_mask).sum(dim=-1)
                        seq_advantages = (advantages * response_mask).sum(dim=-1)
                        # Check correlation: positive scores should have positive advantages
                        pos_score_mask = seq_scores > 0
                        neg_score_mask = seq_scores < 0
                        if pos_score_mask.any():
                            avg_adv_for_pos = seq_advantages[pos_score_mask].mean().item()
                            print(f"[DEBUG] Step {self.global_steps}: Avg advantage for CORRECT (+1): {avg_adv_for_pos:.4f}", flush=True)
                        if neg_score_mask.any():
                            avg_adv_for_neg = seq_advantages[neg_score_mask].mean().item()
                            print(f"[DEBUG] Step {self.global_steps}: Avg advantage for INCORRECT (-1): {avg_adv_for_neg:.4f}", flush=True)
                        print(f"[DEBUG] Step {self.global_steps}: Score dist: +1={pos_score_mask.sum().item()}, -1={neg_score_mask.sum().item()}", flush=True)

                        # Show first 2 examples with token-level detail
                        for ex_idx in range(min(2, scores.shape[0])):
                            resp_mask = response_mask[ex_idx]
                            resp_len = int(resp_mask.sum().item())
                            if resp_len == 0:
                                continue
                            # Find response start (first 1 in mask after padding)
                            resp_start = (resp_mask.cumsum(0) == 1).nonzero(as_tuple=True)[0][0].item()
                            resp_end = resp_start + resp_len

                            ex_tokens = input_ids[ex_idx, resp_start:resp_end].tolist()
                            ex_scores = scores[ex_idx, :resp_len].tolist()
                            ex_advs = advantages[ex_idx, :resp_len].tolist()

                            # Decode tokens
                            try:
                                decoded = self.tokenizer.decode(ex_tokens)
                                # Find where score is non-zero (should be last token)
                                nonzero_score_idx = [i for i, s in enumerate(ex_scores) if abs(s) > 0.01]
                                seq_score = seq_scores[ex_idx].item()
                                seq_adv = seq_advantages[ex_idx].item()

                                print(f"[DEBUG] Example {ex_idx}: score={seq_score:.1f}, total_adv={seq_adv:.4f}", flush=True)
                                print(f"[DEBUG] Example {ex_idx}: Response (last 100 chars): ...{decoded[-100:]}", flush=True)
                                if nonzero_score_idx:
                                    for idx in nonzero_score_idx[-3:]:  # Last 3 non-zero scores
                                        tok = self.tokenizer.decode([ex_tokens[idx]])
                                        print(f"[DEBUG]   Token[{idx}]='{tok}' score={ex_scores[idx]:.2f} adv={ex_advs[idx]:.4f}", flush=True)
                            except Exception as e:
                                print(f"[DEBUG] Example {ex_idx}: decode error: {e}", flush=True)

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

                    if self._force_initial_sync:
                        should_sync = True
                        self._force_initial_sync = False
                        logger.info("Forcing initial weight sync...")

                    batch.meta_info["do_sync"] = should_sync

                    if self._debug_enabled:
                        debug_batch_data(batch, self.global_steps, "before_update", self._debug_output_dir)
                        if self.global_steps in self._debug_save_tensors_at_steps:
                            save_batch_tensors(batch, self.global_steps, self._debug_output_dir)

                    with marked_timer("update_actor", timing_raw, color="magenta"):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if should_sync:
                        self._update_sync_state()

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
