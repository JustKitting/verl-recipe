#!/usr/bin/env python3
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

from . import verl_adapter as _  # noqa: F401

try:
    from environments.gsm8k_server import GSM8kEnv as _UpstreamGSM8kEnv
except ImportError:
    try:
        from atropos.environments.gsm8k_server import GSM8kEnv as _UpstreamGSM8kEnv
    except ImportError:
        raise ImportError(
            "Could not import upstream GSM8kEnv. Please either:\n"
            "1. Clone https://github.com/NousResearch/atropos and add to PYTHONPATH\n"
            "2. Or use the local copy: atropos.environments.gsm8k"
        )

from ..utils import create_verl_adapter

GSM8kEnv = create_verl_adapter(_UpstreamGSM8kEnv, default_tokenizer="Qwen/Qwen3-0.6B")

__all__ = ["GSM8kEnv"]

if __name__ == "__main__":
    GSM8kEnv.cli()
