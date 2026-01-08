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

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def log_section(message: str, char: str = "=", width: int = 60):
    line = char * width
    logger.info(line)
    logger.info(message)
    logger.info(line)


def wait_for_service(url: str, timeout: int = 60, interval: float = 1.0, name: str = None) -> bool:
    name = name or url
    logger.info(f"Waiting for {name}...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                logger.info(f"{name} is ready")
                return True
        except requests.RequestException:
            pass
        time.sleep(interval)

    logger.error(f"{name} not available after {timeout}s")
    return False


def retry_on_failure(attempts: int = 3, min_wait: int = 1, max_wait: int = 10):
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(requests.RequestException),
    )
