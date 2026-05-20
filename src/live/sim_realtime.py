"""
Real-Time Cluster Simulator Bridge.

Wraps the Gymnasium environment K8sSimEnv as a mock PrometheusObserver
and K8sPatchExecutor, allowing LiveClusterAgent to be tested in real-time
or fast-forward simulation.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
import numpy as np

from src.env.k8s_sim import K8sSimEnv
from src.live.live_agent import LiveClusterAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class SimulatedExecutor:
    """Mocks K8sPatchExecutor by stepping the simulation environment."""
    def __init__(self, env: K8sSimEnv) -> None:
        self.env = env
        self.stepped_this_interval = False

    def scale(self, target_replicas: int) -> None:
        # Calculate the replica delta
        delta = target_replicas - self.env.replicas
        # Clip delta to the allowed simulator action space [-3, +3]
        delta = max(-3, min(3, delta))
        action = delta + 3
        
        logger.info(f"[SimulatedExecutor] Scaling to {target_replicas} (delta: {delta})")
        # Step the environment with the action
        self.env.step(action)
        self.stepped_this_interval = True


class SimulatedObserver:
    """Mocks PrometheusObserver by querying the simulator environment state."""
    def __init__(self, env: K8sSimEnv, executor: SimulatedExecutor) -> None:
        self.env = env
        self.executor = executor

    def get_state(self) -> np.ndarray:
        # If the environment was not stepped by the executor in the previous step
        # (i.e. safe_delta was 0, so executor.scale was not called), we must
        # step it with action 3 (delta 0) to progress the simulator time.
        if self.env.step_count > 0 and not self.executor.stepped_this_interval:
            logger.info("[SimulatedObserver] Progressing simulator time (delta: 0)")
            self.env.step(3)  # action 3 maps to delta 0
            
        self.executor.stepped_this_interval = False
        return self.env._build_obs()


def run_simulated_live_run(
    model_path: str | Path,
    duration_steps: int = 120,
    interval_sec: float = 30.0,
    workload_pattern: str = "diurnal",
    run_name: str = "sim_live_run",
) -> None:
    """Run the live agent loop against a simulated cluster.

    Parameters
    ----------
    model_path : str | Path
        Path to the trained PPO model.
    duration_steps : int
        Number of steps to run the experiment.
    interval_sec : float
        Sleep time between steps (set to 0 for instant/fast-forward evaluation).
    workload_pattern : str
        The traffic profile to simulate.
    run_name : str
        Log run name prefix.
    """
    logger.info(f"Starting simulated Phase 2 live run: {run_name}")
    logger.info(f"Model: {model_path}, Workload: {workload_pattern}, Steps: {duration_steps}")

    # Create the simulator environment
    env = K8sSimEnv(workload_pattern=workload_pattern)
    env.reset()

    # Wrap the environment in simulated observer/executor
    executor = SimulatedExecutor(env)
    observer = SimulatedObserver(env, executor)

    # Instantiate the live cluster agent using our mock components
    agent = LiveClusterAgent(
        model_path=model_path,
        run_name=run_name,
        observer=observer,
        executor=executor,
    )

    # Monkeypatch time.sleep inside agent's loop to control the simulation rate
    original_sleep = time.sleep
    if interval_sec != 30.0:
        logger.info(f"Fast-forward mode enabled! Overriding interval from 30s to {interval_sec}s")
        # Define a custom sleep function
        def mock_sleep(seconds: float) -> None:
            if seconds == 30.0:
                original_sleep(interval_sec)
            else:
                original_sleep(seconds)
        time.sleep = mock_sleep

    try:
        # Run the standard live control loop
        agent.run_loop(duration_steps=duration_steps, interval_sec=30)
    finally:
        # Restore original sleep function
        time.sleep = original_sleep
        
    logger.info(f"Simulated live run complete. Logs saved with prefix {run_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 agent loop in a simulated cluster.")
    parser.add_argument("--model", type=str, default="ppo_autoscaler", help="Path to RL model.")
    parser.add_argument("--steps", type=int, default=120, help="Number of 30s steps to run.")
    parser.add_argument("--interval", type=float, default=0.01, help="Sleep interval per step in seconds.")
    parser.add_argument("--workload", type=str, default="diurnal", help="Traffic workload pattern.")
    parser.add_argument("--name", type=str, default="sim_live_run", help="Run name prefix.")
    args = parser.parse_args()

    run_simulated_live_run(
        model_path=args.model,
        duration_steps=args.steps,
        interval_sec=args.interval,
        workload_pattern=args.workload,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
