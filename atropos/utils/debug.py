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

import json
import os
from datetime import datetime

import numpy as np
import torch

from verl import DataProto


def _tensor_stats(t):
    if t is None:
        return {"error": "None"}
    try:
        t = t.float()
        finite_mask = torch.isfinite(t)
        finite_t = t[finite_mask]

        stats = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "nan_count": int((~finite_mask).sum().item()),
            "inf_count": int(torch.isinf(t).sum().item()),
        }

        if finite_t.numel() > 0:
            stats.update({
                "min": float(finite_t.min().item()),
                "max": float(finite_t.max().item()),
                "mean": float(finite_t.mean().item()),
                "std": float(finite_t.std().item()) if finite_t.numel() > 1 else 0.0,
                "median": float(finite_t.median().item()),
                "num_zeros": int((finite_t == 0).sum().item()),
                "num_positive": int((finite_t > 0).sum().item()),
                "num_negative": int((finite_t < 0).sum().item()),
            })
            if finite_t.numel() >= 10:
                for p in [1, 5, 25, 75, 95, 99]:
                    stats[f"p{p}"] = float(torch.quantile(finite_t, p / 100).item())
        else:
            stats["error"] = "No finite values"

        return stats
    except Exception as e:
        return {"error": str(e)}


def debug_batch_data(batch: DataProto, step: int, phase: str, output_dir: str = "./logs"):
    os.makedirs(output_dir, exist_ok=True)

    debug_info = {
        "step": step,
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
        "batch_keys": list(batch.batch.keys()),
        "batch_size": batch.batch.batch_size[0] if batch.batch.batch_size else 0,
    }

    key_tensors = [
        "advantages", "returns", "token_level_rewards", "token_level_scores",
        "old_log_probs", "rollout_log_probs", "ref_log_prob",
        "response_mask", "attention_mask",
    ]

    for key in key_tensors:
        if key in batch.batch.keys():
            debug_info[f"tensor_{key}"] = _tensor_stats(batch.batch[key])

    if "advantages" in batch.batch.keys():
        adv = batch.batch["advantages"]
        resp_mask = batch.batch.get("response_mask", None)

        if resp_mask is not None:
            masked_adv = adv[resp_mask.bool()]
            debug_info["advantages_masked"] = _tensor_stats(masked_adv)

            seq_adv_means = []
            seq_adv_stds = []
            for i in range(adv.shape[0]):
                seq_mask = resp_mask[i].bool()
                if seq_mask.any():
                    seq_adv = adv[i][seq_mask]
                    seq_adv_means.append(seq_adv.mean().item())
                    seq_adv_stds.append(seq_adv.std().item() if seq_adv.numel() > 1 else 0.0)

            if seq_adv_means:
                debug_info["per_seq_adv_means"] = {
                    "min": min(seq_adv_means),
                    "max": max(seq_adv_means),
                    "mean": np.mean(seq_adv_means),
                    "std": np.std(seq_adv_means),
                }
                debug_info["per_seq_adv_stds"] = {
                    "min": min(seq_adv_stds),
                    "max": max(seq_adv_stds),
                    "mean": np.mean(seq_adv_stds),
                }

    if "uid" in batch.non_tensor_batch:
        uids = batch.non_tensor_batch["uid"]
        unique_uids = list(set(uids))
        group_sizes = [list(uids).count(uid) for uid in unique_uids]
        debug_info["grpo_groups"] = {
            "num_groups": len(unique_uids),
            "group_sizes": group_sizes,
            "min_group_size": min(group_sizes) if group_sizes else 0,
            "max_group_size": max(group_sizes) if group_sizes else 0,
            "mean_group_size": np.mean(group_sizes) if group_sizes else 0,
        }

        if "token_level_rewards" in batch.batch.keys() or "token_level_scores" in batch.batch.keys():
            score_key = "token_level_rewards" if "token_level_rewards" in batch.batch.keys() else "token_level_scores"
            scores = batch.batch[score_key].sum(dim=-1)

            group_score_stats = []
            for uid in unique_uids:
                group_mask = np.array([u == uid for u in uids])
                group_scores = scores[group_mask]
                if group_scores.numel() > 0:
                    group_score_stats.append({
                        "size": int(group_scores.numel()),
                        "mean": float(group_scores.mean().item()),
                        "std": float(group_scores.std().item()) if group_scores.numel() > 1 else 0.0,
                        "min": float(group_scores.min().item()),
                        "max": float(group_scores.max().item()),
                    })

            debug_info["grpo_group_scores"] = {
                "num_groups_analyzed": len(group_score_stats),
                "mean_group_mean": np.mean([g["mean"] for g in group_score_stats]) if group_score_stats else 0,
                "mean_group_std": np.mean([g["std"] for g in group_score_stats]) if group_score_stats else 0,
            }

    issues = []
    if "advantages" in batch.batch.keys():
        adv_stats = debug_info.get("tensor_advantages", {})
        if adv_stats.get("nan_count", 0) > 0:
            issues.append(f"NaN in advantages: {adv_stats['nan_count']}")
        if adv_stats.get("inf_count", 0) > 0:
            issues.append(f"Inf in advantages: {adv_stats['inf_count']}")
        if adv_stats.get("std", 1) < 1e-6:
            issues.append(f"Near-zero advantage std: {adv_stats.get('std', 0)}")
        if abs(adv_stats.get("mean", 0)) > 10:
            issues.append(f"Large advantage mean: {adv_stats.get('mean', 0)}")

    if "old_log_probs" in batch.batch.keys():
        lp_stats = debug_info.get("tensor_old_log_probs", {})
        if lp_stats.get("max", -100) > 0:
            issues.append(f"Invalid log_probs > 0: max={lp_stats.get('max', 0)}")
        if lp_stats.get("min", 0) < -100:
            issues.append(f"Very small log_probs: min={lp_stats.get('min', 0)}")

    if "rollout_log_probs" in batch.batch.keys():
        rlp_stats = debug_info.get("tensor_rollout_log_probs", {})
        if rlp_stats.get("max", -100) > 0:
            issues.append(f"Invalid rollout_log_probs > 0: max={rlp_stats.get('max', 0)}")

    debug_info["potential_issues"] = issues

    log_file = os.path.join(output_dir, f"grpo_debug_step_{step:06d}_{phase}.json")
    with open(log_file, "w") as f:
        json.dump(debug_info, f, indent=2, default=str)

    print(f"\n[DEBUG] Step {step} - {phase}: batch_size={debug_info.get('batch_size', 'N/A')}")
    if "tensor_advantages" in debug_info:
        adv = debug_info["tensor_advantages"]
        print(f"  advantages: mean={adv.get('mean', 'N/A'):.4f}, std={adv.get('std', 'N/A'):.4f}")
    if issues:
        print(f"  ISSUES: {', '.join(issues)}")

    return debug_info


def save_batch_tensors(batch: DataProto, step: int, output_dir: str = "./logs"):
    os.makedirs(output_dir, exist_ok=True)

    num_samples = min(4, batch.batch.batch_size[0] if batch.batch.batch_size else 0)
    if num_samples == 0:
        return

    save_data = {}
    key_tensors = [
        "advantages", "returns", "token_level_rewards", "token_level_scores",
        "old_log_probs", "rollout_log_probs", "ref_log_prob",
        "response_mask", "input_ids", "responses",
    ]

    for key in key_tensors:
        if key in batch.batch.keys():
            save_data[key] = batch.batch[key][:num_samples].cpu().numpy().tolist()

    if "uid" in batch.non_tensor_batch:
        save_data["uid"] = list(batch.non_tensor_batch["uid"][:num_samples])

    tensor_file = os.path.join(output_dir, f"batch_tensors_step_{step:06d}.json")
    with open(tensor_file, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"Saved {num_samples} samples to {tensor_file}")
