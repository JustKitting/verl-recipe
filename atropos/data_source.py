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
import uuid
from typing import Optional

import numpy as np
import requests
import torch
from tensordict import TensorDict

from verl import DataProto

from .utils.http import retry_on_failure
from .utils.tensor import pad_sequences

logger = logging.getLogger(__name__)


class AtroposDataSource:

    VALID_TRUNCATION_POLICIES = ("error", "discard", "warn")

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        pad_token_id: int = 0,
        max_seq_len: int = 2048,
        truncation_policy: str = "error",
    ):
        if truncation_policy not in self.VALID_TRUNCATION_POLICIES:
            raise ValueError(
                f"truncation_policy must be one of {self.VALID_TRUNCATION_POLICIES}, "
                f"got '{truncation_policy}'"
            )
        self.api_url = api_url.rstrip("/")
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.truncation_policy = truncation_policy
        self.registered = False
        self.trainer_uuid = None

    @retry_on_failure(attempts=3, min_wait=2, max_wait=10)
    def register(
        self,
        batch_size: int,
        max_token_len: int,
        wandb_group: str = "atropos_verl",
        wandb_project: str = None,
        checkpoint_dir: str = "./checkpoints",
        save_checkpoint_interval: int = 100,
        starting_step: int = 0,
        num_steps: int = 1000,
    ) -> dict:
        response = requests.post(
            f"{self.api_url}/register",
            json={
                "wandb_group": wandb_group,
                "wandb_project": wandb_project,
                "batch_size": batch_size,
                "max_token_len": max_token_len,
                "checkpoint_dir": checkpoint_dir,
                "save_checkpoint_interval": save_checkpoint_interval,
                "starting_step": starting_step,
                "num_steps": num_steps,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        self.trainer_uuid = data.get("uuid")
        self.registered = True
        logger.info(f"Registered with Atropos API, uuid={self.trainer_uuid}")
        return data

    @retry_on_failure(attempts=3, min_wait=1, max_wait=5)
    def get_batch_raw(self) -> dict:
        response = requests.get(f"{self.api_url}/batch", timeout=30)
        if response.status_code == 500:
            logger.warning("API returned 500 - may not be initialized yet")
            return {"batch": None}
        response.raise_for_status()
        return response.json()

    def get_status(self) -> dict:
        try:
            response = requests.get(f"{self.api_url}/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Status request failed with {response.status_code}")
                return {"current_step": 0, "queue_size": 0}
        except requests.RequestException as e:
            logger.warning(f"Failed to get status: {e}")
            return {"current_step": 0, "queue_size": 0}

    def get_batch(self, timeout: float = 60.0) -> Optional[DataProto]:
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                data = self.get_batch_raw()

                if data.get("batch") is None:
                    time.sleep(0.5)
                    continue

                return self._convert_batch(data["batch"])

            except requests.RequestException as e:
                logger.warning(f"Error fetching batch: {e}")
                time.sleep(1.0)

        return None

    def _convert_batch(self, batch_items: list[dict]) -> DataProto:
        all_tokens = []
        all_masks = []
        all_scores = []
        all_advantages = []
        all_ref_logprobs = []
        all_inference_logprobs = []
        all_uids = []
        has_advantages = False
        has_ref_logprobs = False
        has_inference_logprobs = False

        for item in batch_items:
            group_uid = str(uuid.uuid4())

            for i in range(len(item["tokens"])):
                tokens = item["tokens"][i]
                mask = item["masks"][i]

                if len(mask) != len(tokens):
                    raise ValueError(
                        f"Mask length ({len(mask)}) != tokens length ({len(tokens)})"
                    )

                all_tokens.append(tokens)
                all_masks.append(mask)
                all_scores.append(item["scores"][i])
                all_uids.append(group_uid)

                if item.get("advantages") is not None:
                    has_advantages = True
                    all_advantages.append(item["advantages"][i])

                if item.get("ref_logprobs") is not None:
                    has_ref_logprobs = True
                    all_ref_logprobs.append(item["ref_logprobs"][i])

                inf_lp = item.get("inference_logprobs") or item.get("ref_logprobs")
                if inf_lp is not None:
                    has_inference_logprobs = True
                    all_inference_logprobs.append(inf_lp[i])

        batch_size = len(all_tokens)
        actual_max_len = max(len(t) for t in all_tokens)

        if actual_max_len > self.max_seq_len:
            overlong_indices = [i for i, t in enumerate(all_tokens) if len(t) > self.max_seq_len]
            overlong_count = len(overlong_indices)

            if self.truncation_policy == "error":
                raise ValueError(
                    f"{overlong_count}/{batch_size} sequences exceed max_seq_len "
                    f"({actual_max_len} > {self.max_seq_len}). "
                    "Set truncation_policy='discard' to drop overlong samples."
                )
            elif self.truncation_policy == "discard":
                keep_indices = [i for i in range(batch_size) if i not in overlong_indices]
                if not keep_indices:
                    raise ValueError(f"All {batch_size} sequences exceed max_seq_len ({self.max_seq_len}).")
                logger.warning(f"Discarding {overlong_count}/{batch_size} overlong sequences.")
                all_tokens = [all_tokens[i] for i in keep_indices]
                all_masks = [all_masks[i] for i in keep_indices]
                all_scores = [all_scores[i] for i in keep_indices]
                all_uids = [all_uids[i] for i in keep_indices]
                if has_advantages:
                    all_advantages = [all_advantages[i] for i in keep_indices]
                if has_ref_logprobs:
                    all_ref_logprobs = [all_ref_logprobs[i] for i in keep_indices]
                if has_inference_logprobs:
                    all_inference_logprobs = [all_inference_logprobs[i] for i in keep_indices]
                batch_size = len(all_tokens)
                actual_max_len = max(len(t) for t in all_tokens)
            else:
                logger.warning(f"Truncating {overlong_count}/{batch_size} sequences to {self.max_seq_len}.")

        max_len = min(actual_max_len, self.max_seq_len)

        # Compute prompt and response lengths from masks
        prompt_lengths = torch.zeros(batch_size, dtype=torch.long)
        response_lengths = torch.zeros(batch_size, dtype=torch.long)
        for i, mask in enumerate(all_masks):
            seq_len = min(len(mask), max_len)
            found_resp = False
            for j in range(seq_len):
                if mask[j] != -100:
                    if not found_resp:
                        prompt_lengths[i] = j
                        found_resp = True
                    response_lengths[i] += 1
            if not found_resp:
                prompt_lengths[i] = seq_len

        if (response_lengths == 0).any():
            bad_indices = (response_lengths == 0).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(f"Samples {bad_indices} have no response tokens")

        max_prompt_len = max(prompt_lengths.max().item(), 1)
        max_response_len = max(response_lengths.max().item(), 1)

        # Build LEFT-padded prompts (VeRL convention: padding on the left)
        prompts = torch.full((batch_size, max_prompt_len), self.pad_token_id, dtype=torch.long)
        for i in range(batch_size):
            pl = int(prompt_lengths[i].item())
            if pl > 0:
                prompts[i, max_prompt_len - pl:] = torch.tensor(all_tokens[i][:pl], dtype=torch.long)

        # Build RIGHT-padded responses
        responses = torch.full((batch_size, max_response_len), self.pad_token_id, dtype=torch.long)
        for i in range(batch_size):
            pl = int(prompt_lengths[i].item())
            rl = int(response_lengths[i].item())
            if rl > 0:
                responses[i, :rl] = torch.tensor(all_tokens[i][pl:pl + rl], dtype=torch.long)

        # Concatenate: input_ids = [left-padded prompt | right-padded response]
        input_ids = torch.cat([prompts, responses], dim=1)
        total_seq_len = max_prompt_len + max_response_len

        # Build attention_mask (0 for padding, 1 for real tokens)
        attention_mask = torch.zeros(batch_size, total_seq_len, dtype=torch.long)
        for i in range(batch_size):
            pl = int(prompt_lengths[i].item())
            rl = int(response_lengths[i].item())
            attention_mask[i, max_prompt_len - pl : max_prompt_len + rl] = 1

        # Build position_ids (0-indexed from first real token)
        position_ids = attention_mask.cumsum(dim=1) - 1
        position_ids = position_ids.clamp(min=0).long()

        response_only_mask = torch.zeros(batch_size, max_response_len, dtype=torch.float32)
        token_level_scores_response = torch.zeros(batch_size, max_response_len, dtype=torch.float32)
        for i in range(batch_size):
            resp_len = int(response_lengths[i].item())
            response_only_mask[i, :resp_len] = 1.0
            if resp_len > 0:
                token_level_scores_response[i, resp_len - 1] = all_scores[i]

        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_only_mask,
            "token_level_scores": token_level_scores_response,
            "prompts": prompts,
            "responses": responses,
        }

        if has_advantages and len(all_advantages) == batch_size:
            advantages_full = pad_sequences(all_advantages, max_len, 0.0)
            advantages_response = torch.zeros(batch_size, max_response_len, dtype=torch.float32)
            for i in range(batch_size):
                resp_len = int(response_lengths[i].item())
                prompt_len = int(prompt_lengths[i].item())
                advantages_response[i, :resp_len] = advantages_full[i, prompt_len:prompt_len + resp_len]
            batch_dict["atropos_advantages"] = advantages_response

        if has_ref_logprobs and len(all_ref_logprobs) == batch_size:
            ref_logprobs = pad_sequences(all_ref_logprobs, max_len, 0.0)
            batch_dict["ref_log_prob"] = ref_logprobs

        if has_inference_logprobs and len(all_inference_logprobs) == batch_size:
            inference_logprobs_full = pad_sequences(all_inference_logprobs, max_len, 0.0)
            rollout_log_probs = torch.zeros(batch_size, max_response_len, dtype=torch.float32)
            for i in range(batch_size):
                resp_len = int(response_lengths[i].item())
                prompt_len = int(prompt_lengths[i].item())
                rollout_log_probs[i, :resp_len] = inference_logprobs_full[i, prompt_len:prompt_len + resp_len]

            if (rollout_log_probs > 0).any():
                invalid_count = (rollout_log_probs > 0).sum().item()
                logger.warning(f"Invalid logprobs > 0 detected ({invalid_count} values)")
            batch_dict["rollout_log_probs"] = rollout_log_probs

        tensor_dict = TensorDict(batch_dict, batch_size=[batch_size])
        uids = np.array(all_uids, dtype=object)
        batch_seqlens = response_only_mask.sum(dim=1).long().tolist()

        # Final validation - ensure every sample has valid response tokens
        per_sample_valid = response_only_mask.sum(dim=1)
        if (per_sample_valid == 0).any():
            bad_idx = (per_sample_valid == 0).nonzero(as_tuple=True)[0].tolist()
            logger.error(f"Samples {bad_idx} have empty response_mask after processing")
            # Filter out bad samples
            good_mask = per_sample_valid > 0
            if not good_mask.any():
                logger.error("All samples have empty response_mask - returning None")
                return None
            good_idx = good_mask.nonzero(as_tuple=True)[0]
            tensor_dict = tensor_dict[good_idx]
            uids = uids[good_idx.numpy()]
            batch_seqlens = [batch_seqlens[i] for i in good_idx.tolist()]
            batch_size = len(good_idx)
            logger.warning(f"Filtered to {batch_size} valid samples")

        return DataProto(
            batch=tensor_dict,
            non_tensor_batch={"uid": uids},
            meta_info={
                "source": "atropos",
                "global_token_num": batch_seqlens,
            },
        )
