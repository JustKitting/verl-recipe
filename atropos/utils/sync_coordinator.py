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
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set

import ray

logger = logging.getLogger(__name__)

COORDINATOR_NAME = "atropos_sync_coordinator"


class RequestState(Enum):
    ACTIVE = "active"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"


@dataclass
class TrackedRequest:
    request_id: str
    generation: int
    start_time: float
    state: RequestState = RequestState.ACTIVE


@dataclass
class GenerationInfo:
    generation: int
    start_time: float
    cancelled: bool = False
    request_ids: Set[str] = field(default_factory=set)


@ray.remote
class SyncCoordinator:
    """Coordinates weight sync between FSDP workers and Atropos inference requests."""

    def __init__(self, request_timeout: float = 300.0, cleanup_interval: float = 30.0):
        self._current_generation = 0
        self._sync_in_progress_count = 0
        self._drain_pending = False
        self._requests: Dict[str, TrackedRequest] = {}
        self._generation_info: Dict[int, GenerationInfo] = {}
        self._inference_urls = []
        self._request_timeout = request_timeout
        self._cleanup_interval = cleanup_interval
        self._last_cleanup_time = time.time()

        self._generation_info[0] = GenerationInfo(generation=0, start_time=time.time())
        logger.info(f"SyncCoordinator initialized: request_timeout={request_timeout}s")

    def set_inference_urls(self, urls: list[str]):
        self._inference_urls = urls
        logger.info(f"Stored {len(urls)} inference URL(s)")

    def get_inference_urls(self) -> list[str]:
        return self._inference_urls

    def register_request(self) -> dict:
        self._maybe_cleanup_timed_out()

        request_id = str(uuid.uuid4())
        generation = self._current_generation

        request = TrackedRequest(
            request_id=request_id,
            generation=generation,
            start_time=time.time(),
            state=RequestState.ACTIVE,
        )
        self._requests[request_id] = request

        if generation in self._generation_info:
            self._generation_info[generation].request_ids.add(request_id)

        should_proceed = self._sync_in_progress_count == 0

        return {
            "request_id": request_id,
            "generation": generation,
            "should_proceed": should_proceed,
        }

    def complete_request(self, request_id: str) -> dict:
        if request_id not in self._requests:
            logger.warning(f"complete_request called for unknown request_id={request_id}")
            return {
                "was_stale": False,
                "was_cancelled": False,
                "generation": -1,
                "current_generation": self._current_generation,
            }

        request = self._requests[request_id]
        was_cancelled = request.state == RequestState.CANCELLED
        was_stale = request.generation < self._current_generation

        request.state = RequestState.COMPLETED

        if request.generation in self._generation_info:
            self._generation_info[request.generation].request_ids.discard(request_id)

        del self._requests[request_id]

        return {
            "was_stale": was_stale,
            "was_cancelled": was_cancelled,
            "generation": request.generation,
            "current_generation": self._current_generation,
        }

    def check_request_valid(self, request_id: str) -> dict:
        if request_id not in self._requests:
            return {"valid": False, "reason": "unknown", "generation": -1}

        request = self._requests[request_id]

        if request.state == RequestState.CANCELLED:
            return {"valid": False, "reason": "cancelled", "generation": request.generation}

        if request.state == RequestState.TIMED_OUT:
            return {"valid": False, "reason": "timed_out", "generation": request.generation}

        elapsed = time.time() - request.start_time
        if elapsed > self._request_timeout:
            request.state = RequestState.TIMED_OUT
            return {"valid": False, "reason": "timed_out", "generation": request.generation}

        return {"valid": True, "reason": None, "generation": request.generation}

    def begin_sync(self) -> dict:
        self._sync_in_progress_count += 1
        previous_generation = self._current_generation

        cancelled_count = 0
        active_count = 0

        for request in self._requests.values():
            if request.state == RequestState.ACTIVE:
                if request.generation <= previous_generation:
                    request.state = RequestState.CANCELLED
                    cancelled_count += 1
                else:
                    active_count += 1

        if previous_generation in self._generation_info:
            self._generation_info[previous_generation].cancelled = True

        logger.info(
            f"Sync started: gen={previous_generation}, "
            f"cancelled={cancelled_count}, active={active_count}"
        )

        return {
            "previous_generation": previous_generation,
            "new_generation": previous_generation + 1,
            "cancelled_requests": cancelled_count,
            "active_requests": active_count,
        }

    def end_sync(self) -> dict:
        self._sync_in_progress_count = max(0, self._sync_in_progress_count - 1)

        cleaned = 0
        if self._sync_in_progress_count == 0:
            self._current_generation += 1

            self._generation_info[self._current_generation] = GenerationInfo(
                generation=self._current_generation,
                start_time=time.time(),
            )

            gens_to_remove = []
            for gen in list(self._generation_info.keys()):
                if gen < self._current_generation - 1:
                    gen_info = self._generation_info[gen]
                    if len(gen_info.request_ids) == 0:
                        gens_to_remove.append(gen)

            for gen in gens_to_remove:
                del self._generation_info[gen]
                cleaned += 1

            logger.info(f"Sync ended: new_generation={self._current_generation}")

        return {
            "generation": self._current_generation,
            "cleaned_generations": cleaned,
        }

    def begin_drain(self) -> dict:
        self._drain_pending = True
        active_count = self._get_active_count()

        return {
            "drain_pending": True,
            "active_requests": active_count,
            "generation": self._current_generation,
        }

    def end_drain(self) -> dict:
        self._drain_pending = False

        return {
            "drain_pending": False,
            "generation": self._current_generation,
        }

    def is_drain_pending(self) -> bool:
        return self._drain_pending

    def get_drain_status(self) -> dict:
        self._maybe_cleanup_timed_out()

        active_by_gen = {}
        active_by_state = {s.value: 0 for s in RequestState}
        oldest_age = 0.0
        total_active = 0
        now = time.time()

        for request in self._requests.values():
            active_by_state[request.state.value] += 1

            if request.state == RequestState.ACTIVE:
                total_active += 1
                gen = request.generation
                active_by_gen[gen] = active_by_gen.get(gen, 0) + 1
                age = now - request.start_time
                oldest_age = max(oldest_age, age)

        return {
            "is_drained": total_active == 0,
            "total_active": total_active,
            "active_by_generation": active_by_gen,
            "active_by_state": active_by_state,
            "oldest_request_age": oldest_age,
        }

    def force_drain(self) -> dict:
        force_drained = 0
        by_gen = {}
        by_age = {"<10s": 0, "10-60s": 0, "60-300s": 0, ">300s": 0}
        now = time.time()

        requests_to_remove = []
        for request_id, request in self._requests.items():
            if request.state == RequestState.ACTIVE:
                force_drained += 1
                gen = request.generation
                by_gen[gen] = by_gen.get(gen, 0) + 1

                age = now - request.start_time
                if age < 10:
                    by_age["<10s"] += 1
                elif age < 60:
                    by_age["10-60s"] += 1
                elif age < 300:
                    by_age["60-300s"] += 1
                else:
                    by_age[">300s"] += 1

                request.state = RequestState.TIMED_OUT
                requests_to_remove.append(request_id)

        for request_id in requests_to_remove:
            request = self._requests.get(request_id)
            if request and request.generation in self._generation_info:
                self._generation_info[request.generation].request_ids.discard(request_id)
            if request_id in self._requests:
                del self._requests[request_id]

        logger.warning(f"Force-drained {force_drained} requests: by_gen={by_gen}, by_age={by_age}")

        return {
            "force_drained_count": force_drained,
            "by_generation": by_gen,
            "by_age": by_age,
        }

    def is_sync_in_progress(self) -> bool:
        return self._sync_in_progress_count > 0 or self._drain_pending

    def is_drained(self) -> bool:
        return self._get_active_count() == 0

    def get_in_flight_count(self) -> int:
        return self._get_active_count()

    def increment_in_flight(self) -> str:
        result = self.register_request()
        return result["request_id"]

    def decrement_in_flight(self, request_id: str = None):
        if request_id:
            self.complete_request(request_id)
        else:
            oldest_id = None
            oldest_time = float('inf')
            for rid, req in self._requests.items():
                if req.state == RequestState.ACTIVE and req.start_time < oldest_time:
                    oldest_time = req.start_time
                    oldest_id = rid
            if oldest_id:
                self.complete_request(oldest_id)

    def reset_in_flight(self) -> dict:
        return self.force_drain()

    def _get_active_count(self) -> int:
        return sum(1 for r in self._requests.values() if r.state == RequestState.ACTIVE)

    def _maybe_cleanup_timed_out(self):
        now = time.time()
        if now - self._last_cleanup_time < self._cleanup_interval:
            return

        self._last_cleanup_time = now
        timed_out = []

        for request_id, request in self._requests.items():
            if request.state == RequestState.ACTIVE:
                elapsed = now - request.start_time
                if elapsed > self._request_timeout:
                    request.state = RequestState.TIMED_OUT
                    timed_out.append(request_id)

        if timed_out:
            logger.warning(f"Cleanup: timed out {len(timed_out)} stale requests")
            for request_id in timed_out:
                request = self._requests.get(request_id)
                if request and request.generation in self._generation_info:
                    self._generation_info[request.generation].request_ids.discard(request_id)
                if request_id in self._requests:
                    del self._requests[request_id]


def get_coordinator():
    return ray.get_actor(COORDINATOR_NAME)


def create_coordinator(request_timeout: float = 300.0, cleanup_interval: float = 30.0):
    try:
        return ray.get_actor(COORDINATOR_NAME)
    except ValueError:
        return SyncCoordinator.options(
            name=COORDINATOR_NAME,
            lifetime="detached",
        ).remote(request_timeout=request_timeout, cleanup_interval=cleanup_interval)
