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

import ray

logger = logging.getLogger(__name__)

COORDINATOR_NAME = "atropos_sync_coordinator"


@ray.remote
class SyncCoordinator:
    """Ray actor for coordinating weight sync between FSDP workers and Atropos inference requests."""

    def __init__(self):
        self._sync_in_progress = False
        self._in_flight_count = 0
        self._inference_urls = []

    def set_inference_urls(self, urls: list[str]):
        self._inference_urls = urls
        logger.info(f"Stored {len(urls)} inference URL(s)")

    def get_inference_urls(self) -> list[str]:
        return self._inference_urls

    def begin_sync(self):
        self._sync_in_progress = True
        if self._in_flight_count < 0:
            logger.warning(f"Resetting negative in_flight_count={self._in_flight_count} to 0")
            self._in_flight_count = 0
        logger.info(f"Sync started, in_flight_count={self._in_flight_count}")

    def reset_in_flight(self):
        old_count = self._in_flight_count
        self._in_flight_count = 0
        logger.info(f"Reset in_flight_count from {old_count} to 0")

    def end_sync(self):
        self._sync_in_progress = False
        logger.info("Sync ended, requests unblocked")

    def is_sync_in_progress(self) -> bool:
        return self._sync_in_progress

    def increment_in_flight(self):
        self._in_flight_count += 1

    def decrement_in_flight(self):
        self._in_flight_count -= 1

    def get_in_flight_count(self) -> int:
        return self._in_flight_count

    def is_drained(self) -> bool:
        return self._in_flight_count == 0


def get_coordinator():
    """Get the named sync coordinator actor."""
    return ray.get_actor(COORDINATOR_NAME)


def create_coordinator():
    """Create the sync coordinator actor (idempotent)."""
    try:
        return ray.get_actor(COORDINATOR_NAME)
    except ValueError:
        return SyncCoordinator.options(
            name=COORDINATOR_NAME,
            lifetime="detached",
        ).remote()
