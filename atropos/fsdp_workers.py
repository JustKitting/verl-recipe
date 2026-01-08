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
import time

import ray
from verl import DataProto
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_torch_device
from verl.utils.debug import log_gpu_memory_usage

from .utils.sync_coordinator import get_coordinator

logger = logging.getLogger(__name__)

DRAIN_TIMEOUT = 300  # Max seconds to wait for requests to drain


class AtroposActorRolloutRefWorker(AsyncActorRolloutRefWorker):

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        output = super().update_actor(data)

        get_torch_device().empty_cache()

        do_sync = data.meta_info.get("do_sync", False)

        if do_sync and self._is_actor and hasattr(self, 'rollout') and self.rollout is not None:
            sync_metrics = self._sync_weights_to_inference()
            if "metrics" not in output.meta_info:
                output.meta_info["metrics"] = {}
            output.meta_info["metrics"].update(sync_metrics)

        return output

    async def _wait_for_drain(self) -> dict:
        metrics = {}

        try:
            coordinator = get_coordinator()
        except Exception as e:
            logger.warning(f"Could not get sync coordinator: {e}")
            return metrics

        drain_start = time.time()

        ray.get(coordinator.begin_sync.remote())
        in_flight = ray.get(coordinator.get_in_flight_count.remote())
        logger.info(f"Sync started, waiting for {in_flight} in-flight requests to complete...")

        while not ray.get(coordinator.is_drained.remote()):
            elapsed = time.time() - drain_start
            if elapsed > DRAIN_TIMEOUT:
                remaining = ray.get(coordinator.get_in_flight_count.remote())
                logger.warning(f"Drain timeout after {DRAIN_TIMEOUT}s, {remaining} requests still in-flight - resetting")
                ray.get(coordinator.reset_in_flight.remote())
                break
            await asyncio.sleep(0.1)

        drain_time = time.time() - drain_start
        metrics["sync/drain_time"] = drain_time
        logger.info(f"Drained in {drain_time:.2f}s")

        return metrics

    async def _end_sync(self):
        try:
            coordinator = get_coordinator()
            ray.get(coordinator.end_sync.remote())
        except Exception as e:
            logger.warning(f"Could not signal sync end: {e}")

    def _sync_weights_to_inference(self) -> dict:
        metrics = {}

        logger.info("Syncing weights to SGLang rollout engine...")
        sync_start = time.time()
        log_gpu_memory_usage("Before weight sync", logger=logger)

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
        log_gpu_memory_usage("After sync", logger=logger)

        get_torch_device().empty_cache()

        total_sync_time = time.time() - sync_start
        metrics["sync/total_time"] = total_sync_time
        logger.info(f"Weight sync complete in {total_sync_time:.2f}s (drain: {drain_time:.2f}s)")

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
