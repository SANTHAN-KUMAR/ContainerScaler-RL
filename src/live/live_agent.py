"""
LiveClusterAgent — Orchestrator for Phase 2 (Live Deployment).

Wires together the Observer, Agent, Executor, and Logger.
Runs the main control loop every 30 seconds.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from src.agents.agent import ContainerScaleAgent
from src.live.executor import K8sPatchExecutor
from src.live.metrics_logger import MetricsLogger
from src.live.observer import PrometheusObserver
from src.safety.safety_filter import ClusterState

logger = logging.getLogger(__name__)


class LiveClusterAgent:
    """Orchestrates the live cluster scaling loop.

    Parameters
    ----------
    prom_url : str
        Prometheus URL.
    namespace : str
        Kubernetes namespace.
    deployment : str
        Target deployment name.
    model_path : str | Path
        Path to trained PPO model.
    run_name : str
        Prefix for logs.
    """

    def __init__(
        self,
        prom_url: str = "http://localhost:9090",
        namespace: str = "default",
        deployment: str = "podinfo",
        model_path: str | Path = "ppo_autoscaler",
        run_name: str = "live_run",
        observer: Any = None,
        executor: Any = None,
    ) -> None:
        if observer is not None:
            self.observer = observer
        else:
            self.observer = PrometheusObserver(
                prom_url=prom_url, namespace=namespace, deployment=deployment
            )
            
        if executor is not None:
            self.executor = executor
        else:
            self.executor = K8sPatchExecutor(
                namespace=namespace, deployment=deployment
            )
        
        # Load the agent. Passing model_path="ensemble" loads the EnsembleMetaAgent.
        try:
            m_path = str(model_path).lower()
            if m_path == "hpa":
                logger.info("HPA Baseline selected. Bypassing RL model.")
                raise Exception("Forced HPA")
            elif m_path == "ensemble":
                from src.agents.ensemble_agent import EnsembleMetaAgent
                self.agent = EnsembleMetaAgent()
                logger.info("Loaded EnsembleMetaAgent as live controller.")
            else:
                self.agent = ContainerScaleAgent(model_path=model_path)
            self.has_rl = True
        except Exception as e:
            logger.error("Failed to load RL model: %s. Running in HPA-only mode.", e)
            self.agent = None
            self.has_rl = False
            from src.agents.hpa_baseline import RealisticHPA
            from src.safety.safety_filter import SafetyFilter
            self.hpa = RealisticHPA()
            self.safety = SafetyFilter()

        self.logger = MetricsLogger(prefix=run_name)
        
        self.logger.save_meta({
            "prom_url": prom_url,
            "namespace": namespace,
            "deployment": deployment,
            "model_path": str(model_path),
            "run_name": run_name,
            "interval_seconds": 30,
        })

    def run_loop(self, duration_steps: int = 120, interval_sec: int = 30) -> None:
        """Run the control loop.

        Parameters
        ----------
        duration_steps : int
            Number of steps to run.
        interval_sec : int
            Wait time between steps.
        """
        logger.info(f"Starting live control loop for {duration_steps} steps.")

        for step in range(duration_steps):
            start_time = time.time()
            logger.info(f"--- Step {step}/{duration_steps} ---")

            try:
                # 1. Observe
                obs = self.observer.get_state()
                state = ClusterState.from_obs(obs)

                # 2. Decide & filter
                if self.has_rl:
                    decision = self.agent.decide_with_info(obs, step)
                    safe_delta = decision["delta"]
                    proposed_delta = decision["proposed_delta"]
                    source = decision["source"]
                else:
                    proposed_delta = self.hpa.act(state)
                    safe_delta = self.safety.check(state, proposed_delta, step)
                    source = "hpa"

                # 3. Act
                if safe_delta != 0:
                    target = state.replicas + safe_delta
                    self.executor.scale(target)
                else:
                    logger.info("No scale action needed.")

                # 4. Log
                self.logger.log(
                    step=step,
                    state=state,
                    proposed_delta=proposed_delta,
                    safe_delta=safe_delta,
                    source=source,
                )

            except Exception as e:
                logger.error(f"Error in control loop step {step}: {e}")

            # 5. Sleep and poll for live UI updates
            elapsed = time.time() - start_time
            sleep_time = max(0.0, interval_sec - elapsed)
            if sleep_time > 0:
                end_sleep = time.time() + sleep_time
                while time.time() < end_sleep:
                    # Sleep in 2-second chunks and re-log current metrics for the UI
                    time_to_sleep = min(2.0, end_sleep - time.time())
                    if time_to_sleep > 0:
                        time.sleep(time_to_sleep)
                    try:
                        fast_obs = self.observer.get_state()
                        fast_state = ClusterState.from_obs(fast_obs)
                        self.logger.log(
                            step=step,
                            state=fast_state,
                            proposed_delta=proposed_delta,
                            safe_delta=safe_delta,
                            source=source,
                        )
                    except Exception:
                        pass
            else:
                logger.warning(f"Control loop taking longer than {interval_sec}s!")

        logger.info("Live control loop completed.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--prom", default="http://localhost:9090", help="Prometheus URL")
    parser.add_argument("--namespace", default="default", help="K8s namespace")
    parser.add_argument("--deployment", default="frontend", help="Target deployment (Online Boutique frontend)")
    parser.add_argument("--model", default="ensemble", help="Model path or 'ensemble' for EnsembleMetaAgent")
    parser.add_argument("--steps", type=int, default=120, help="Number of steps")
    parser.add_argument("--interval", type=int, default=30, help="Interval between steps in seconds")
    parser.add_argument("--name", default="live_run", help="Run name prefix for logs")
    
    args = parser.parse_args()
    
    agent = LiveClusterAgent(
        prom_url=args.prom,
        namespace=args.namespace,
        deployment=args.deployment,
        model_path=args.model,
        run_name=args.name,
    )
    agent.run_loop(duration_steps=args.steps, interval_sec=args.interval)


if __name__ == "__main__":
    main()
