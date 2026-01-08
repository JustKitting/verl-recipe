#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import atexit
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

import requests

from .utils.http import log_section, wait_for_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    model_path: str = "Qwen/Qwen3-0.6B"
    atropos_api_port: int = 8000
    total_steps: int = 1000
    num_gpus: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-6
    use_wandb: bool = True
    wandb_project: str = "atropos_verl"
    experiment_name: str = "atropos_grpo"
    log_dir: str = "./logs"


class ServiceManager:

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.processes: list[subprocess.Popen] = []
        self._inference_url = None
        self._setup_signal_handlers()
        self._setup_log_dir()

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.cleanup)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)

    def _setup_log_dir(self):
        os.makedirs(self.config.log_dir, exist_ok=True)
        logger.info(f"Logs will be saved to: {self.config.log_dir}")

    def cleanup(self):
        logger.info("Cleaning up processes...")
        for proc in reversed(self.processes):
            if proc.poll() is None:
                logger.info(f"Terminating PID {proc.pid}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing PID {proc.pid}")
                    proc.kill()
        self.processes.clear()

        logger.info("Stopping Ray cluster...")
        try:
            result = subprocess.run(
                ["ray", "stop", "--force"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info("Ray cluster stopped.")
            else:
                logger.warning(f"Ray stop returned {result.returncode}: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning("Ray stop timed out")
        except FileNotFoundError:
            logger.debug("Ray CLI not found, skipping ray stop")

        logger.info("Cleanup complete.")

    def start_atropos_api(self) -> bool:
        log_section("Starting Atropos Trajectory API...")

        log_file = os.path.join(self.config.log_dir, "atropos_api.log")

        cmd = [
            sys.executable,
            "-m",
            "atroposlib.cli.run_api",
            "--port",
            str(self.config.atropos_api_port),
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        with open(log_file, "w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

        self.processes.append(proc)
        logger.info(f"PID: {proc.pid}, Log: {log_file}")

        api_url = f"http://localhost:{self.config.atropos_api_port}/"
        return wait_for_service(api_url, timeout=30, name="Atropos API")

    def start_verl_trainer(self) -> bool:
        log_section("Starting VeRL Trainer...")

        log_file = os.path.join(self.config.log_dir, "verl_trainer.log")
        verl_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        cmd = [
            sys.executable,
            "-m",
            "atropos.main",
            f"atropos.api_url=http://localhost:{self.config.atropos_api_port}",
            "atropos.batch_timeout=120.0",
            f"actor_rollout_ref.model.path={self.config.model_path}",
            "actor_rollout_ref.rollout.n=4",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={self.config.num_gpus}",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
            f"actor_rollout_ref.actor.optim.lr={self.config.learning_rate}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={self.config.batch_size}",
            "actor_rollout_ref.actor.use_kl_loss=True",
            "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            f"data.train_batch_size={self.config.batch_size}",
            "algorithm.adv_estimator=grpo",
            "algorithm.norm_adv_by_std_in_grpo=True",
            f"trainer.n_gpus_per_node={self.config.num_gpus}",
            f"trainer.total_training_steps={self.config.total_steps}",
            f"trainer.project_name={self.config.wandb_project}",
            f"trainer.experiment_name={self.config.experiment_name}",
        ]

        if self.config.use_wandb:
            cmd.append('trainer.logger=["console","wandb"]')
        else:
            cmd.append('trainer.logger=["console"]')

        logger.info(f"Command: {' '.join(cmd[:5])}... (truncated)")

        env = os.environ.copy()
        env["PYTHONPATH"] = verl_repo_root + ":" + env.get("PYTHONPATH", "")

        with open(log_file, "w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

        self.processes.append(proc)
        logger.info(f"PID: {proc.pid}, Log: {log_file}")

        time.sleep(5)
        if proc.poll() is not None:
            logger.error("VeRL trainer process failed to start")
            return False

        logger.info("VeRL trainer started")
        return True

    def wait_for_inference_url(self, timeout: int = 300) -> str:
        logger.info("Waiting for VeRL trainer to register with Trajectory API...")
        start_time = time.time()
        api_url = f"http://localhost:{self.config.atropos_api_port}"

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{api_url}/", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    url = data.get("inference_url")
                    if url:
                        self._inference_url = url.replace("/v1", "").rstrip("/")
                        logger.info(f"Discovered inference URL: {self._inference_url}")
                        return self._inference_url
            except requests.RequestException:
                pass
            time.sleep(2)

        raise RuntimeError(f"Timeout waiting for inference URL after {timeout}s")

    def run(self) -> int:
        log_section("Atropos + VeRL Training Launcher")
        logger.info(f"Model: {self.config.model_path}")
        logger.info(f"Steps: {self.config.total_steps}")
        logger.info(f"GPUs: {self.config.num_gpus}")

        if not self.start_atropos_api():
            logger.error("Failed to start Atropos API")
            return 1

        if not self.start_verl_trainer():
            logger.error("Failed to start VeRL trainer")
            return 1

        try:
            inference_url = self.wait_for_inference_url()
        except RuntimeError as e:
            logger.error(f"Failed to discover inference URL: {e}")
            return 1

        log_section("Services started")
        logger.info(f"  Trajectory API:  http://localhost:{self.config.atropos_api_port}")
        logger.info(f"  SGLang Server:   {inference_url}")
        logger.info(f"  Monitor: tail -f {self.config.log_dir}/verl_trainer.log")

        trainer_proc = self.processes[1]
        try:
            exit_code = trainer_proc.wait()
            logger.info(f"Training completed with exit code: {exit_code}")
            return exit_code
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            return 0


def parse_args() -> ServiceConfig:
    parser = argparse.ArgumentParser(
        description="Launch Atropos + VeRL training services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m atropos.launch_services --model Qwen/Qwen2.5-1.5B-Instruct --steps 1000
  python -m atropos.launch_services --gpus 2 --no-wandb
        """,
    )

    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--atropos-port", type=int, default=8000, help="Trajectory API port")
    parser.add_argument("--wandb-project", default="atropos_verl", help="W&B project name")
    parser.add_argument("--experiment-name", default=None, help="Experiment name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")
    parser.add_argument("--log-dir", default="./logs", help="Log directory")

    args = parser.parse_args()

    if args.experiment_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"atropos_grpo_{timestamp}"

    return ServiceConfig(
        model_path=args.model,
        atropos_api_port=args.atropos_port,
        total_steps=args.steps,
        num_gpus=args.gpus,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        experiment_name=args.experiment_name,
        log_dir=args.log_dir,
    )


def main():
    config = parse_args()
    manager = ServiceManager(config)
    sys.exit(manager.run())


if __name__ == "__main__":
    main()
