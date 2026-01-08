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
import logging
import os

import ray
from contextlib import asynccontextmanager

from .sync_coordinator import get_coordinator

logger = logging.getLogger(__name__)


_sglang_patch_applied = False


SYNC_WAIT_TIMEOUT = 120.0


class VeRLManagedServerWrapper:
    """Wrapper that coordinates inference requests with VeRL weight syncs."""

    def __init__(self, managed_server):
        self._managed = managed_server
        self._coordinator = None
        self._init_coordinator()

    def _init_coordinator(self):
        try:
            if not ray.is_initialized():
                namespace = os.environ.get("VERL_RAY_NAMESPACE", "verl")
                ray.init(address="auto", namespace=namespace, ignore_reinit_error=True)
            self._coordinator = get_coordinator()
        except Exception as e:
            logger.debug(f"Coordinator not available: {e}")

    async def _wait_for_sync(self):
        if self._coordinator is None:
            return
        start = asyncio.get_event_loop().time()
        while ray.get(self._coordinator.is_sync_in_progress.remote()):
            if asyncio.get_event_loop().time() - start > SYNC_WAIT_TIMEOUT:
                logger.warning(f"Sync wait timeout after {SYNC_WAIT_TIMEOUT}s, proceeding anyway")
                break
            await asyncio.sleep(0.05)

    def _increment_in_flight(self):
        if self._coordinator is not None:
            try:
                ray.get(self._coordinator.increment_in_flight.remote())
            except Exception as e:
                logger.warning(f"Failed to increment in_flight: {e}")

    def _decrement_in_flight(self):
        if self._coordinator is not None:
            try:
                ray.get(self._coordinator.decrement_in_flight.remote())
            except Exception as e:
                logger.warning(f"Failed to decrement in_flight: {e}")

    async def chat_completion(self, **kwargs):
        await self._wait_for_sync()
        self._increment_in_flight()
        try:
            return await self._managed.chat_completion(**kwargs)
        finally:
            self._decrement_in_flight()

    async def completion(self, **kwargs):
        await self._wait_for_sync()
        self._increment_in_flight()
        try:
            return await self._managed.completion(**kwargs)
        finally:
            self._decrement_in_flight()

    def get_state(self):
        return self._managed.get_state()

    def reset(self):
        if hasattr(self._managed, 'reset'):
            self._managed.reset()


def apply_managed_server_patch():
    try:
        from atroposlib.envs.server_handling.server_manager import ServerManager

        original_managed_server = ServerManager.managed_server

        @asynccontextmanager
        async def wrapped_managed_server(self, tokenizer=None):
            async with original_managed_server(self, tokenizer) as managed:
                wrapper = VeRLManagedServerWrapper(managed)
                try:
                    yield wrapper
                finally:
                    wrapper.reset()

        ServerManager.managed_server = wrapped_managed_server
        print("[sglang_patch] Wrapped managed_server with VeRL coordination")
    except Exception as e:
        print(f"[sglang_patch] Failed to patch ServerManager: {e}")


def apply_sglang_server_patch():
    """Patch aiohttp to inject logprob_start_len=0 into /generate requests."""
    import aiohttp

    original_post = aiohttp.ClientSession._request

    async def patched_request(self, method, url, **kwargs):
        if method.upper() == 'POST' and '/generate' in str(url) and 'json' in kwargs:
            kwargs['json']['logprob_start_len'] = 0
        return await original_post(self, method, url, **kwargs)

    aiohttp.ClientSession._request = patched_request
    print("[sglang_patch] Patched aiohttp to inject logprob_start_len=0")


def apply_sglang_patch():
    global _sglang_patch_applied
    if _sglang_patch_applied:
        return
    _sglang_patch_applied = True

    apply_managed_server_patch()
    apply_sglang_server_patch()
