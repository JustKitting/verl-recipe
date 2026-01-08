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
from typing import List, Tuple, Type, TypeVar

import ray
from atroposlib.envs.base import BaseEnv, BaseEnvConfig
from atroposlib.envs.server_handling.server_baseline import APIServerConfig

from .sync_coordinator import get_coordinator


logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseEnv)


def get_verl_server_configs(model_name: str) -> List[APIServerConfig]:
    if not ray.is_initialized():
        namespace = os.environ.get("VERL_RAY_NAMESPACE", "verl")
        ray.init(address="auto", namespace=namespace, ignore_reinit_error=True)
    coordinator = get_coordinator()
    server_urls = ray.get(coordinator.get_inference_urls.remote())

    if not server_urls:
        raise ValueError("No inference URLs available from sync coordinator.")

    servers = []
    for url in server_urls:
        base_url = url if url.endswith("/v1") else url.rstrip("/") + "/v1"
        servers.append(
            APIServerConfig(
                base_url=base_url,
                model_name=model_name,
                server_type="sglang",
                api_key="x",
                timeout=1200,
                num_max_requests_at_once=512,
                num_requests_for_eval=256,
            )
        )

    print(f"[verl_adapter] Configured {len(servers)} server(s), model: {model_name}")
    return servers


def create_verl_adapter(
    env_class: Type[T],
    default_tokenizer: str = "Qwen/Qwen3-0.6B",
) -> Type[T]:
    class VeRLAdapter(env_class):
        _verl_server_configs: List[APIServerConfig] = None
        _verl_model_name: str = default_tokenizer

        @classmethod
        def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
            try:
                env_config, _ = super(VeRLAdapter, cls).config_init()
            except Exception:
                env_config = cls.env_config_cls() if hasattr(cls, 'env_config_cls') else BaseEnvConfig()

            env_config.tokenizer_name = cls._verl_model_name
            server_configs = get_verl_server_configs(cls._verl_model_name)
            cls._verl_server_configs = server_configs
            return env_config, server_configs

        def __init__(self, config, server_configs, *args, **kwargs):
            super().__init__(config, server_configs, *args, **kwargs)

    VeRLAdapter.__name__ = f"VeRL{env_class.__name__}"
    VeRLAdapter.__qualname__ = f"VeRL{env_class.__name__}"
    return VeRLAdapter
