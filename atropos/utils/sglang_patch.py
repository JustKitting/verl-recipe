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
import os

import ray

from .sync_coordinator import get_coordinator

logger = logging.getLogger(__name__)


_sglang_patch_applied = False


def apply_sglang_server_patch():
    """Patch aiohttp to inject logprob_start_len=0 into /generate requests AND coordinate with sync."""
    import aiohttp
    import asyncio
    import time
    import os

    original_post = aiohttp.ClientSession._request
    _pid = os.getpid()
    _request_counter = [0]  # mutable counter
    _coordinator = [None]  # mutable holder for coordinator

    def _get_coordinator():
        """Lazily get the coordinator."""
        if _coordinator[0] is None:
            try:
                _coordinator[0] = get_coordinator()
            except Exception as e:
                print(f"[HTTP_REQ pid={_pid}] Coordinator not available: {e}")
        return _coordinator[0]

    async def patched_request(self, method, url, **kwargs):
        _request_counter[0] += 1
        req_num = _request_counter[0]
        url_str = str(url)

        # Log ALL HTTP requests
        if '/generate' in url_str:
            json_data = kwargs.get('json', {})
            input_ids = json_data.get('input_ids', [])
            input_len = len(input_ids) if isinstance(input_ids, list) else 'unknown'
            sampling = json_data.get('sampling_params', {})
            max_tokens = sampling.get('max_new_tokens', json_data.get('max_new_tokens', 'unknown'))
            kwargs['json']['logprob_start_len'] = 0

            # === SYNC COORDINATION ===
            coordinator = _get_coordinator()
            req_id = None
            start = time.time()

            # Retry loop for stale responses
            MAX_STALE_RETRIES = 3
            for attempt in range(MAX_STALE_RETRIES):
                req_id = None
                gen = None

                try:
                    if coordinator is not None:
                        # Wait for any in-progress sync to complete
                        wait_count = 0
                        while ray.get(coordinator.is_sync_in_progress.remote()):
                            wait_count += 1
                            if wait_count % 100 == 1:
                                print(f"[HTTP_REQ pid={_pid}] #{req_num} Waiting for sync... ({wait_count * 0.05:.1f}s)")
                            await asyncio.sleep(0.05)
                        if wait_count > 0:
                            print(f"[HTTP_REQ pid={_pid}] #{req_num} Done waiting for sync after {wait_count * 0.05:.1f}s")

                        # Register request so drain waits for us
                        reg_result = ray.get(coordinator.register_request.remote())
                        req_id = reg_result["request_id"]
                        gen = reg_result["generation"]
                        print(f"[HTTP_REQ pid={_pid}] #{req_num} Registered {req_id[:8]}, gen={gen}, input_len={input_len}, attempt={attempt+1}")

                        # Check if sync started while we were registering (race condition)
                        if not reg_result["should_proceed"]:
                            wait_count2 = 0
                            while ray.get(coordinator.is_sync_in_progress.remote()):
                                wait_count2 += 1
                                if wait_count2 % 100 == 1:
                                    print(f"[HTTP_REQ pid={_pid}] #{req_num} Waiting (race)... ({wait_count2 * 0.05:.1f}s)")
                                await asyncio.sleep(0.05)
                            if wait_count2 > 0:
                                print(f"[HTTP_REQ pid={_pid}] #{req_num} Done waiting (race) after {wait_count2 * 0.05:.1f}s")
                    else:
                        print(f"[HTTP_REQ pid={_pid}] #{req_num} POST {url_str} input_len={input_len} (no coordinator)")

                    # Make the actual request
                    result = await original_post(self, method, url, **kwargs)
                    elapsed = time.time() - start

                    # Check staleness BEFORE returning
                    should_retry = False
                    if coordinator is not None and req_id is not None:
                        try:
                            complete_result = ray.get(coordinator.complete_request.remote(req_id))
                            was_stale = complete_result.get("was_stale", False)
                            was_cancelled = complete_result.get("was_cancelled", False)
                            current_gen = complete_result.get("current_generation", gen)

                            if was_stale or was_cancelled:
                                reason = "STALE" if was_stale else "CANCELLED"
                                print(f"[HTTP_REQ pid={_pid}] #{req_num} {reason} response (gen={gen} -> current={current_gen}), attempt {attempt+1}/{MAX_STALE_RETRIES}, retrying...")
                                should_retry = True
                                req_id = None  # Already completed, don't complete again in finally
                            else:
                                print(f"[HTTP_REQ pid={_pid}] #{req_num} COMPLETED fresh (gen={gen}) in {elapsed:.2f}s")
                                req_id = None  # Already completed
                        except Exception as e:
                            print(f"[HTTP_REQ pid={_pid}] #{req_num} Error completing request: {e}")
                    else:
                        print(f"[HTTP_REQ pid={_pid}] #{req_num} COMPLETED in {elapsed:.2f}s (no coordinator)")

                    if not should_retry:
                        return result
                    # Otherwise continue loop to retry

                finally:
                    # Complete request if not already completed
                    if coordinator is not None and req_id is not None:
                        try:
                            ray.get(coordinator.complete_request.remote(req_id))
                        except Exception as e:
                            print(f"[HTTP_REQ pid={_pid}] #{req_num} Error completing request in finally: {e}")

            # Exhausted retries - return last result anyway with warning
            print(f"[HTTP_REQ pid={_pid}] #{req_num} WARNING: Exhausted {MAX_STALE_RETRIES} retries, returning potentially stale response")
            return result

        elif '/update_weights' in url_str:
            print(f"[HTTP_REQ pid={_pid}] #{req_num} POST {url_str} (weight update)")
            start = time.time()
            result = await original_post(self, method, url, **kwargs)
            elapsed = time.time() - start
            print(f"[HTTP_REQ pid={_pid}] #{req_num} weight update COMPLETED in {elapsed:.2f}s")
            return result
        else:
            return await original_post(self, method, url, **kwargs)

    aiohttp.ClientSession._request = patched_request
    print("[sglang_patch] Patched aiohttp to inject logprob_start_len=0 and coordinate with sync")


def apply_verl_rollout_sync_patch():
    """Patch SGLangHttpServer.generate to check with SyncCoordinator.

    This ensures verl's training rollout requests are blocked during weight sync.
    NOTE: verl rollout uses SGLangHttpServer.generate() via Ray, NOT AsyncHttpServerAdapter.

    IMPORTANT: Ray actors run in separate processes. We must patch __init__ to apply
    the generate patch INSIDE the actor's process when it's instantiated.
    """
    import functools

    MAX_STALE_RETRIES = 3

    try:
        from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer

        # SGLangHttpServer is decorated with @ray.remote, which wraps the class.
        # The actual class is stored in __ray_actor_class__.
        actual_class = SGLangHttpServer.__ray_actor_class__
        original_init = actual_class.__init__

        @functools.wraps(original_init)
        def patched_init(self, *args, **kwargs):
            # Call original __init__ first
            original_init(self, *args, **kwargs)

            # Now patch this instance's generate method
            import asyncio
            import os as _os
            import time as _time
            import ray

            _pid = _os.getpid()
            print(f"[GENERATE_PATCH pid={_pid}] Patching generate method on SGLangHttpServer instance")

            original_generate = self.generate

            async def synced_generate(
                prompt_ids,
                sampling_params,
                request_id,
                image_data=None,
                video_data=None,
            ):
                _outer_start = _time.time()

                for attempt in range(MAX_STALE_RETRIES):
                    _start = _time.time()
                    req_id = None
                    coordinator = None
                    generation = None

                    # === SYNC COORDINATION: Wait and register ===
                    try:
                        from atropos.utils.sync_coordinator import get_coordinator
                        coordinator = get_coordinator()

                        # Wait if sync is in progress
                        wait_count = 0
                        while ray.get(coordinator.is_sync_in_progress.remote()):
                            wait_count += 1
                            if wait_count % 100 == 1:
                                print(f"[GENERATE pid={_pid}] Waiting for sync to complete... (waited {wait_count * 0.05:.1f}s)")
                            await asyncio.sleep(0.05)

                        if wait_count > 0:
                            print(f"[GENERATE pid={_pid}] Done waiting for sync after {wait_count * 0.05:.1f}s")

                        # Register this request so drain knows about us
                        reg_result = ray.get(coordinator.register_request.remote())
                        req_id = reg_result["request_id"]
                        generation = reg_result["generation"]
                        print(f"[GENERATE pid={_pid}] Registered request {req_id[:8]}, gen={generation}")

                        # Wait again if sync started while we were registering
                        if not reg_result["should_proceed"]:
                            wait_count2 = 0
                            while ray.get(coordinator.is_sync_in_progress.remote()):
                                wait_count2 += 1
                                if wait_count2 % 100 == 1:
                                    print(f"[GENERATE pid={_pid}] Waiting again for sync (race condition)... (waited {wait_count2 * 0.05:.1f}s)")
                                await asyncio.sleep(0.05)
                            if wait_count2 > 0:
                                print(f"[GENERATE pid={_pid}] Done waiting (race) after {wait_count2 * 0.05:.1f}s")

                    except Exception as e:
                        print(f"[GENERATE pid={_pid}] ERROR: Could not register with coordinator: {e}")

                    # === ACTUAL GENERATION ===
                    output = None
                    generation_error = None
                    try:
                        _gen_start = _time.time()
                        output = await original_generate(
                            prompt_ids=prompt_ids,
                            sampling_params=sampling_params,
                            request_id=request_id,
                            image_data=image_data,
                            video_data=video_data,
                        )
                        _gen_time = _time.time() - _gen_start
                        print(f"[GENERATE pid={_pid}] Generate completed in {_gen_time:.2f}s")
                    except Exception as e:
                        generation_error = e

                    # === SYNC COORDINATION: Complete and CHECK STALENESS ===
                    should_retry = False
                    if coordinator is not None and req_id is not None:
                        try:
                            complete_result = ray.get(coordinator.complete_request.remote(req_id))
                            was_stale = complete_result.get("was_stale", False)
                            was_cancelled = complete_result.get("was_cancelled", False)
                            current_gen = complete_result.get("current_generation", generation)

                            if was_stale or was_cancelled:
                                reason = "stale" if was_stale else "cancelled"
                                print(
                                    f"[GENERATE pid={_pid}] Response {reason} "
                                    f"(gen={generation} -> current={current_gen}), "
                                    f"attempt {attempt + 1}/{MAX_STALE_RETRIES}, retrying..."
                                )
                                should_retry = True
                            else:
                                print(f"[GENERATE pid={_pid}] Completed {req_id[:8]}, fresh (gen={generation}), total={_time.time()-_start:.2f}s")
                        except Exception as e:
                            print(f"[GENERATE pid={_pid}] ERROR completing request: {e}")

                    # If generation failed, raise the error
                    if generation_error is not None:
                        raise generation_error

                    # If response is fresh, return it
                    if not should_retry:
                        return output

                    # Otherwise, loop and retry with fresh weights

                # Exhausted retries
                print(
                    f"[GENERATE pid={_pid}] Failed to get fresh response after {MAX_STALE_RETRIES} retries "
                    f"(total time={_time.time()-_outer_start:.2f}s)"
                )
                raise RuntimeError(
                    f"Failed to generate fresh response after {MAX_STALE_RETRIES} retries - "
                    f"weights keep changing during generation"
                )

            # Replace instance method
            self.generate = synced_generate
            print(f"[GENERATE_PATCH pid={_pid}] Successfully patched generate method with stale rejection")

        actual_class.__init__ = patched_init
        print("[sglang_patch] Patched SGLangHttpServer.__init__ to apply generate sync on instantiation")

    except Exception as e:
        print(f"[sglang_patch] Failed to patch SGLangHttpServer: {e}")


def apply_weight_sync_debug_logging():
    """Add debug logging to trace weight sync issues."""
    import functools

    # Patch ServerAdapter.update_weights to log entry/exit
    try:
        from verl.workers.rollout.sglang_rollout import sglang_rollout

        original_update_weights = sglang_rollout.ServerAdapter.update_weights

        @functools.wraps(original_update_weights)
        async def logged_update_weights(self, weights, **kwargs):
            import time
            start = time.time()
            print(f"[weight_sync_debug] ServerAdapter.update_weights() called")
            print(f"[weight_sync_debug] kwargs: {kwargs}")
            print(f"[weight_sync_debug] device_mesh infer_tp local_rank: {self.device_mesh['infer_tp'].get_local_rank()}")

            # Count tensors
            weights_list = list(weights)
            print(f"[weight_sync_debug] Number of weight tensors: {len(weights_list)}")
            if weights_list:
                names = [n for n, _ in weights_list[:5]]
                print(f"[weight_sync_debug] First 5 tensor names: {names}")

            # Call original with the list (generator was consumed)
            result = await original_update_weights(self, iter(weights_list), **kwargs)

            elapsed = time.time() - start
            print(f"[weight_sync_debug] ServerAdapter.update_weights() completed in {elapsed:.2f}s, result: {result}")
            return result

        sglang_rollout.ServerAdapter.update_weights = logged_update_weights
        print("[weight_sync_debug] Patched ServerAdapter.update_weights with logging")
    except Exception as e:
        print(f"[weight_sync_debug] Failed to patch ServerAdapter.update_weights: {e}")

    # Patch AsyncHttpServerAdapter.update_weights_from_tensor to log HTTP calls
    try:
        from verl.workers.rollout.sglang_rollout import http_server_engine

        original_http_update = http_server_engine.AsyncHttpServerAdapter.update_weights_from_tensor

        @functools.wraps(original_http_update)
        async def logged_http_update(self, req):
            import time
            start = time.time()
            print(f"[weight_sync_debug] HTTP update_weights_from_tensor called")
            print(f"[weight_sync_debug] num serialized tensors: {len(req.serialized_named_tensors)}")
            print(f"[weight_sync_debug] load_format: {req.load_format}")
            print(f"[weight_sync_debug] server: {self.server_args.host}:{self.server_args.port}")

            result = await original_http_update(self, req)

            elapsed = time.time() - start
            print(f"[weight_sync_debug] HTTP update completed in {elapsed:.2f}s")
            print(f"[weight_sync_debug] HTTP result: {result}")
            return result

        http_server_engine.AsyncHttpServerAdapter.update_weights_from_tensor = logged_http_update
        print("[weight_sync_debug] Patched AsyncHttpServerAdapter.update_weights_from_tensor with logging")
    except Exception as e:
        print(f"[weight_sync_debug] Failed to patch HTTP adapter: {e}")

    # Patch sgl_update_weights to log its calls
    try:
        from verl.workers.rollout.sglang_rollout import sglang_rollout
        from sglang.srt.weight_sync import utils as sgl_weight_utils

        original_sgl_update = sgl_weight_utils.update_weights

        @functools.wraps(original_sgl_update)
        async def logged_sgl_update(engine, params_batch, device_mesh_key, device_mesh, load_format=None):
            import time
            start = time.time()
            infer_tp_rank = device_mesh[device_mesh_key].get_local_rank()
            print(f"[weight_sync_debug] sgl_update_weights called, infer_tp_rank={infer_tp_rank}")
            print(f"[weight_sync_debug] params_batch size: {len(params_batch)}")
            print(f"[weight_sync_debug] engine type: {type(engine).__name__}")

            result = await original_sgl_update(engine, params_batch, device_mesh_key, device_mesh, load_format)

            elapsed = time.time() - start
            print(f"[weight_sync_debug] sgl_update_weights completed in {elapsed:.2f}s, result: {result}")
            return result

        # Patch both the module and the import in sglang_rollout
        sgl_weight_utils.update_weights = logged_sgl_update
        sglang_rollout.sgl_update_weights = logged_sgl_update
        print("[weight_sync_debug] Patched sgl_update_weights with logging")
    except Exception as e:
        print(f"[weight_sync_debug] Failed to patch sgl_update_weights: {e}")


def apply_server_side_logging():
    """Server-side logging disabled - was causing errors."""
    print("[sglang_patch] Server-side logging disabled (use client-side HTTP_REQ logs instead)")


def apply_async_llm_server_manager_patch():
    """Patch AsyncLLMServerManager.generate to coordinate with weight sync.

    This is the class used by AgentLoopWorker to make generate calls.
    By patching it, we ensure all generate requests:
    1. Wait for any in-progress weight sync to complete
    2. Register with SyncCoordinator so drain waits for them
    3. Complete tracking and CHECK STALENESS
    4. Retry if response was generated with stale weights
    """
    import asyncio
    import functools
    import os
    import time
    from uuid import uuid4

    import ray

    MAX_STALE_RETRIES = 3

    try:
        from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager

        original_generate = AsyncLLMServerManager.generate
        _pid = os.getpid()

        @functools.wraps(original_generate)
        async def synced_generate(self, request_id, *, prompt_ids, sampling_params,
                                   image_data=None, video_data=None):
            _outer_start = time.time()

            for attempt in range(MAX_STALE_RETRIES):
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
                            print(f"[GENERATE pid={_pid}] Waiting for sync... ({_wait_count * 0.05:.1f}s)")
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
                                print(f"[GENERATE pid={_pid}] Waiting (race)... ({_wait_count2 * 0.05:.1f}s)")
                            await asyncio.sleep(0.05)
                        if _wait_count2 > 0:
                            print(f"[GENERATE pid={_pid}] Done waiting (race) after {_wait_count2 * 0.05:.1f}s")

                except Exception as e:
                    print(f"[GENERATE pid={_pid}] Sync coordinator unavailable: {e}")

                # === ACTUAL GENERATION (call original) ===
                output = None
                generation_error = None
                try:
                    output = await original_generate(
                        self, request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        image_data=image_data,
                        video_data=video_data,
                    )
                except Exception as e:
                    generation_error = e

                # === SYNC COORDINATION: Complete and CHECK STALENESS ===
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
                                f"attempt {attempt + 1}/{MAX_STALE_RETRIES}, retrying..."
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
                f"[GENERATE pid={_pid}] Failed to get fresh response after {MAX_STALE_RETRIES} retries "
                f"(total time={time.time()-_outer_start:.2f}s)"
            )
            raise RuntimeError(
                f"Failed to generate fresh response after {MAX_STALE_RETRIES} retries - "
                f"weights keep changing during generation"
            )

        AsyncLLMServerManager.generate = synced_generate
        print("[sglang_patch] Patched AsyncLLMServerManager.generate with sync coordination and stale rejection")

    except Exception as e:
        print(f"[sglang_patch] Failed to patch AsyncLLMServerManager.generate: {e}")


def apply_sglang_patch():
    global _sglang_patch_applied
    if _sglang_patch_applied:
        return
    _sglang_patch_applied = True

    apply_sglang_server_patch()
    apply_verl_rollout_sync_patch()
    apply_weight_sync_debug_logging()
    apply_server_side_logging()
    apply_async_llm_server_manager_patch()
