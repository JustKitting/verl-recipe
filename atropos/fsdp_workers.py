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

import asyncio
import concurrent.futures
import logging
import os
import time

import ray
import torch
import torch.distributed as dist
from verl import DataProto
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_torch_device
from verl.utils.debug import log_gpu_memory_usage

from .utils.sync_coordinator import get_coordinator

logger = logging.getLogger(__name__)


def _get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def _get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def _barrier():
    if dist.is_initialized() and _get_world_size() > 1:
        dist.barrier()


def _broadcast_dict(data: dict, src: int = 0) -> dict:
    if not dist.is_initialized() or _get_world_size() == 1:
        return data

    rank = _get_rank()

    if rank == src:
        import pickle
        serialized = pickle.dumps(data)
        size_tensor = torch.tensor([len(serialized)], dtype=torch.long, device="cuda")
    else:
        size_tensor = torch.tensor([0], dtype=torch.long, device="cuda")

    dist.broadcast(size_tensor, src=src)
    size = size_tensor.item()

    if rank == src:
        data_tensor = torch.tensor(list(serialized), dtype=torch.uint8, device="cuda")
    else:
        data_tensor = torch.empty(size, dtype=torch.uint8, device="cuda")

    dist.broadcast(data_tensor, src=src)

    if rank != src:
        import pickle
        serialized = bytes(data_tensor.cpu().tolist())
        data = pickle.loads(serialized)

    return data


DRAIN_TIMEOUT = 300
DRAIN_LOG_INTERVAL = 10


class AtroposActorRolloutRefWorker(AsyncActorRolloutRefWorker):

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        if "response_mask" in data.batch:
            rm = data.batch["response_mask"]
            if not rm.any():
                raise ValueError(
                    f"Batch has no valid response tokens (response_mask all zeros). "
                    f"Shape: {rm.shape}, sum: {rm.sum().item()}"
                )

        output = super().update_actor(data)
        get_torch_device().empty_cache()

        do_sync = data.meta_info.get("do_sync", False)
        if do_sync and self._is_actor and hasattr(self, 'rollout') and self.rollout is not None:
            sync_metrics = self._sync_weights_to_inference()
            if "metrics" not in output.meta_info:
                output.meta_info["metrics"] = {}
            output.meta_info["metrics"].update(sync_metrics)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sync_weights_only(self) -> dict:
        if self._is_actor and hasattr(self, 'rollout') and self.rollout is not None:
            sync_metrics = self._sync_weights_to_inference()
            return {"metrics": sync_metrics}
        return {"metrics": {}}

    async def _wait_for_drain(self) -> dict:
        metrics = {}
        rank = _get_rank()

        _barrier()
        drain_start = time.time()

        if rank == 0:
            try:
                coordinator = get_coordinator()
            except Exception as e:
                logger.warning(f"Could not get sync coordinator: {e}")
                sync_info = {"previous_generation": -1, "new_generation": -1, "cancelled_requests": 0, "active_requests": 0, "error": str(e)}
            else:
                sync_info = ray.get(coordinator.begin_sync.remote())
                logger.info(
                    f"Sync started: gen {sync_info['previous_generation']} -> {sync_info['new_generation']}, "
                    f"cancelled={sync_info['cancelled_requests']}, active={sync_info['active_requests']}"
                )
        else:
            sync_info = {}

        sync_info = _broadcast_dict(sync_info, src=0)

        if "error" not in sync_info:
            metrics["sync/previous_generation"] = sync_info["previous_generation"]
            metrics["sync/new_generation"] = sync_info["new_generation"]
            metrics["sync/cancelled_requests"] = sync_info["cancelled_requests"]

        if rank == 0:
            last_log_time = drain_start
            try:
                coordinator = get_coordinator()
                while True:
                    drain_status = ray.get(coordinator.get_drain_status.remote())

                    if drain_status["is_drained"]:
                        break

                    elapsed = time.time() - drain_start

                    if elapsed - (last_log_time - drain_start) >= DRAIN_LOG_INTERVAL:
                        last_log_time = time.time()
                        logger.info(
                            f"Drain progress: {drain_status['total_active']} active, "
                            f"by_gen={drain_status['active_by_generation']}, "
                            f"oldest={drain_status['oldest_request_age']:.1f}s"
                        )

                    if elapsed > DRAIN_TIMEOUT:
                        logger.warning(f"Drain timeout after {DRAIN_TIMEOUT}s with {drain_status['total_active']} active requests")
                        force_result = ray.get(coordinator.force_drain.remote())
                        metrics["sync/force_drained_count"] = force_result["force_drained_count"]
                        logger.warning(f"Force-drained {force_result['force_drained_count']} requests")
                        break

                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Drain loop error: {e}")

        drain_time = time.time() - drain_start
        _barrier()

        metrics["sync/drain_time"] = drain_time
        if rank == 0:
            logger.info(f"Drained in {drain_time:.2f}s")

        return metrics

    async def _end_sync(self):
        rank = _get_rank()

        _barrier()

        if rank == 0:
            try:
                coordinator = get_coordinator()
                end_result = ray.get(coordinator.end_sync.remote())
                logger.info(
                    f"Sync ended: new_generation={end_result['generation']}, "
                    f"cleaned_generations={end_result['cleaned_generations']}"
                )
            except Exception as e:
                logger.warning(f"Could not signal sync end: {e}")

        _barrier()

    def _sync_weights_to_inference(self) -> dict:
        metrics = {}
        rank = _get_rank()

        logger.info(f"Rank {rank}: Syncing weights to SGLang rollout engine...")
        sync_start = time.time()
        log_gpu_memory_usage(f"Rank {rank}: Before weight sync", logger=logger)

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._run_sync_coroutines)
                drain_time, rollout_time, trainer_time, drain_metrics = future.result(timeout=600)
                metrics.update(drain_metrics)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                drain_start = time.time()
                drain_metrics = loop.run_until_complete(self._wait_for_drain())
                drain_time = time.time() - drain_start
                metrics.update(drain_metrics)

                try:
                    rollout_start = time.time()
                    loop.run_until_complete(self.rollout_mode())
                    rollout_time = time.time() - rollout_start

                    trainer_start = time.time()
                    loop.run_until_complete(self.trainer_mode())
                    trainer_time = time.time() - trainer_start
                finally:
                    loop.run_until_complete(self._end_sync())
            finally:
                loop.close()

        metrics["sync/drain_time"] = drain_time
        metrics["sync/rollout_mode_time"] = rollout_time
        metrics["sync/trainer_mode_time"] = trainer_time
        log_gpu_memory_usage(f"Rank {rank}: After sync", logger=logger)

        get_torch_device().empty_cache()

        total_sync_time = time.time() - sync_start
        metrics["sync/total_time"] = total_sync_time
        logger.info(f"Rank {rank}: Weight sync complete in {total_sync_time:.2f}s (drain: {drain_time:.2f}s)")

        return metrics

    def _run_sync_coroutines(self) -> tuple[float, float, float, dict]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            drain_start = time.time()
            drain_metrics = loop.run_until_complete(self._wait_for_drain())
            drain_time = time.time() - drain_start

            try:
                rollout_start = time.time()
                loop.run_until_complete(self.rollout_mode())
                rollout_time = time.time() - rollout_start

                trainer_start = time.time()
                loop.run_until_complete(self.trainer_mode())
                trainer_time = time.time() - trainer_start
            finally:
                loop.run_until_complete(self._end_sync())

            return drain_time, rollout_time, trainer_time, drain_metrics
        finally:
            loop.close()
