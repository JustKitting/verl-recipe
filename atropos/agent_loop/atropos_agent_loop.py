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

"""
Atropos Agent Loop with weight sync coordination.

This module provides custom AgentLoop classes that coordinate inference requests
with weight sync operations. During weight sync, generate requests are blocked
to prevent using stale weights.
"""

import asyncio
import logging
import os
import time
from typing import Any, Optional
from uuid import uuid4

import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopWorkerBase,
    AsyncLLMServerManager,
)
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.workers.rollout.replica import TokenOutput

from atropos.utils.sync_coordinator import get_coordinator

logger = logging.getLogger(__name__)
_pid = os.getpid()


class AtroposAsyncLLMServerManager(AsyncLLMServerManager):
    """
    AsyncLLMServerManager with weight sync coordination.

    Wraps generate() to:
    1. Wait for any in-progress weight sync to complete
    2. Register the request with SyncCoordinator so drain knows about it
    3. Complete the request when done and CHECK STALENESS
    4. Retry if response was generated with stale weights
    """

    MAX_STALE_RETRIES = 3

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens with sync coordination and stale response rejection."""
        _outer_start = time.time()

        for attempt in range(self.MAX_STALE_RETRIES):
            _start = time.time()
            _req_id = None
            _coordinator = None
            _generation = None

            # === SYNC COORDINATION: Wait and register ===
            try:
                _coordinator = get_coordinator()

                # Wait if sync is in progress
                _wait_count = 0
                while ray.get(_coordinator.is_sync_in_progress.remote()):
                    _wait_count += 1
                    if _wait_count % 100 == 1:
                        print(f"[GENERATE pid={_pid}] Waiting for sync... (waited {_wait_count * 0.05:.1f}s)")
                    await asyncio.sleep(0.05)

                if _wait_count > 0:
                    print(f"[GENERATE pid={_pid}] Done waiting for sync after {_wait_count * 0.05:.1f}s")

                # Register request so drain waits for us
                _result = ray.get(_coordinator.register_request.remote())
                _req_id = _result["request_id"]
                _generation = _result["generation"]
                print(f"[GENERATE pid={_pid}] Registered {_req_id[:8]}, gen={_generation}")

                # Wait again if sync started while registering (race condition)
                if not _result["should_proceed"]:
                    _wait_count2 = 0
                    while ray.get(_coordinator.is_sync_in_progress.remote()):
                        _wait_count2 += 1
                        if _wait_count2 % 100 == 1:
                            print(f"[GENERATE pid={_pid}] Waiting (race)... (waited {_wait_count2 * 0.05:.1f}s)")
                        await asyncio.sleep(0.05)
                    if _wait_count2 > 0:
                        print(f"[GENERATE pid={_pid}] Done waiting (race) after {_wait_count2 * 0.05:.1f}s")

            except Exception as e:
                print(f"[GENERATE pid={_pid}] Sync coordinator unavailable: {e}")
                # Fall through to do generation without coordination

            # === ACTUAL GENERATION ===
            output = None
            generation_error = None
            try:
                server = self._choose_server(request_id)
                output = await server.generate.remote(
                    request_id=uuid4().hex,  # use new request_id for each turn
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=image_data,
                    video_data=video_data,
                )
            except Exception as e:
                generation_error = e

            # === SYNC COORDINATION: Complete request and CHECK STALENESS ===
            should_retry = False
            if _coordinator is not None and _req_id is not None:
                try:
                    complete_result = ray.get(_coordinator.complete_request.remote(_req_id))
                    was_stale = complete_result.get("was_stale", False)
                    was_cancelled = complete_result.get("was_cancelled", False)
                    current_gen = complete_result.get("current_generation", _generation)

                    if was_stale or was_cancelled:
                        reason = "stale" if was_stale else "cancelled"
                        print(
                            f"[GENERATE pid={_pid}] Response {reason} "
                            f"(gen={_generation} -> current={current_gen}), "
                            f"attempt {attempt + 1}/{self.MAX_STALE_RETRIES}, retrying..."
                        )
                        should_retry = True
                    else:
                        print(f"[GENERATE pid={_pid}] Completed {_req_id[:8]}, fresh (gen={_generation}), total={time.time()-_start:.2f}s")
                except Exception as e:
                    print(f"[GENERATE pid={_pid}] Error completing request: {e}")

            # If generation failed, raise the error
            if generation_error is not None:
                raise generation_error

            # If response is fresh, return it
            if not should_retry:
                return output

            # Otherwise, loop and retry with fresh weights

        # Exhausted retries
        print(
            f"[GENERATE pid={_pid}] Failed to get fresh response after {self.MAX_STALE_RETRIES} retries "
            f"(total time={time.time()-_outer_start:.2f}s)"
        )
        raise RuntimeError(
            f"Failed to generate fresh response after {self.MAX_STALE_RETRIES} retries - "
            f"weights keep changing during generation"
        )


class AtroposAgentLoopWorkerBase(AgentLoopWorkerBase):
    """Agent loop worker using AtroposAsyncLLMServerManager for sync coordination."""

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
    ):
        # Call parent init but override server_manager
        super().__init__(config, server_handles, reward_router_address)
        # Replace with our sync-coordinated version
        self.server_manager = AtroposAsyncLLMServerManager(config, server_handles)
        print(f"[AtroposAgentLoopWorker pid={_pid}] Using AtroposAsyncLLMServerManager with sync coordination")


# Wrap with @ray.remote for use as Ray actor
AtroposAgentLoopWorker = ray.remote(AtroposAgentLoopWorkerBase)


class AtroposAgentLoopManager(AgentLoopManager):
    """Agent loop manager using AtroposAgentLoopWorker for sync coordination."""

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rm_resource_pool: RayResourcePool = None,
    ):
        # Set custom worker class before calling parent init
        self.agent_loop_workers_class = AtroposAgentLoopWorker
        super().__init__(config, worker_group, rm_resource_pool)
        print(f"[AtroposAgentLoopManager] Initialized with sync-coordinated workers")
