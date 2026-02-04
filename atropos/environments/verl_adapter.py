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

import argparse
import importlib
import sys
import time
import traceback
from typing import Optional

from ..utils import create_verl_adapter, set_ray_connection

MAX_RETRIES = 10
INITIAL_BACKOFF = 5.0
MAX_BACKOFF = 60.0


def find_env_class(module):
    from atroposlib.envs.base import BaseEnv
    for name in ['Environment', 'Env', 'GSM8kEnv', 'LetterCountingEnv']:
        if hasattr(module, name):
            cls = getattr(module, name)
            if isinstance(cls, type) and issubclass(cls, BaseEnv):
                return cls
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, BaseEnv) and obj is not BaseEnv:
            return obj
    return None


def load_env(env_module: str, env_class_name: Optional[str] = None):
    try:
        module = importlib.import_module(env_module)
        print(f"Loaded from: {env_module}")
    except ImportError as e:
        raise ImportError(f"Could not import '{env_module}': {e}")

    if env_class_name:
        if not hasattr(module, env_class_name):
            raise AttributeError(f"Module {env_module} has no class '{env_class_name}'")
        return getattr(module, env_class_name)

    env_class = find_env_class(module)
    if env_class is None:
        raise AttributeError(f"No BaseEnv subclass in {env_module}. Use --env-class.")
    return env_class


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env-module", required=True)
    parser.add_argument("--env-class", default=None)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--ray-namespace", default="verl")
    parser.add_argument("--ray-address", default="auto")
    args, remaining = parser.parse_known_args()

    set_ray_connection(namespace=args.ray_namespace, address=args.ray_address)

    print(f"Loading: {args.env_module}")
    BaseEnvClass = load_env(args.env_module, args.env_class)
    print(f"Found: {BaseEnvClass.__name__}")

    WrappedEnv = create_verl_adapter(BaseEnvClass, default_tokenizer=args.tokenizer)
    sys.argv = [sys.argv[0]] + remaining

    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            WrappedEnv.cli()
            break
        except SystemExit as e:
            if e.code == 0:
                break
            print(f"[verl_adapter] Environment exited with code {e.code}")
            raise
        except Exception as e:
            print(f"\n[verl_adapter] Environment crashed (attempt {attempt + 1}/{MAX_RETRIES})")
            print(f"[verl_adapter] Error: {type(e).__name__}: {e}")
            traceback.print_exc()

            if attempt + 1 >= MAX_RETRIES:
                print(f"[verl_adapter] Max retries exceeded, giving up")
                raise

            print(f"[verl_adapter] Restarting in {backoff:.1f}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
            print(f"[verl_adapter] Restarting environment...")


if __name__ == "__main__":
    main()
