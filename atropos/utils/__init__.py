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

from .patches import apply_all_patches
from .sglang_patch import apply_sglang_patch

apply_all_patches()
apply_sglang_patch()

from .debug import debug_batch_data, save_batch_tensors
from .http import log_section, retry_on_failure, wait_for_service
from .tensor import pad_sequences
from .env_adapter import create_verl_adapter, get_verl_server_configs

__all__ = [
    "apply_all_patches",
    "apply_sglang_patch",
    "create_verl_adapter",
    "debug_batch_data",
    "get_verl_server_configs",
    "log_section",
    "pad_sequences",
    "retry_on_failure",
    "save_batch_tensors",
    "wait_for_service",
]
