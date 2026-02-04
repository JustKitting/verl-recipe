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
from typing import Callable, List, Optional, Tuple, Type, TypeVar

import ray
import torch
from atroposlib.envs.base import BaseEnv, BaseEnvConfig
from atroposlib.envs.server_handling.server_baseline import APIServerConfig

from .sync_coordinator import get_coordinator


logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseEnv)

_ray_namespace: Optional[str] = None
_ray_address: Optional[str] = None


def set_ray_connection(namespace: str, address: str = "auto"):
    global _ray_namespace, _ray_address
    _ray_namespace = namespace
    _ray_address = address


class VeRLScoreAdapter:

    @staticmethod
    def scores_to_rewards(batch: "DataProto") -> "DataProto":
        if "token_level_scores" in batch.batch.keys():
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
        return batch

    @staticmethod
    def apply_atropos_advantages(batch: "DataProto") -> "DataProto":
        if "atropos_advantages" in batch.batch.keys():
            batch.batch["advantages"] = (
                batch.batch["advantages"] + batch.batch["atropos_advantages"]
            )
        return batch


def compute_advantage_with_score_adapter(
    data: "DataProto",
    compute_advantage_fn: Callable,
    adv_estimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> "DataProto":
    data = compute_advantage_fn(
        data=data,
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
    data = VeRLScoreAdapter.apply_atropos_advantages(data)
    return data


def get_verl_server_configs(model_name: str) -> List[APIServerConfig]:
    if not ray.is_initialized():
        namespace = _ray_namespace or "verl"
        address = _ray_address or "auto"
        ray.init(address=address, namespace=namespace, ignore_reinit_error=True)
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

    logger.info(f"Configured {len(servers)} server(s), model: {model_name}")
    return servers


def create_verl_adapter(
    env_class: Type[T],
    default_tokenizer: str = "Qwen/Qwen2.5-3B",
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
