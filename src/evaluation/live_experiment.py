"""
Experiment 8: HPA vs RL on Live / Simulated Cluster.

Runs comparative experiments between the HPA baseline and the RL agent
under identical traffic profiles, computes performance metrics, and plots
trajectories.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pandas as pd

from src.env.k8s_sim import K8sSimEnv
from src.live.sim_realtime import SimulatedObserver, SimulatedExecutor
from src.live.live_agent import LiveClusterAgent
from src.evaluation.metrics import (
    compute_sla_compliance,
    compute_average_cost,
    compute_churn,
    compute_max_latency,
)
from src.evaluation.visualize import plot_live_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_experiment(
    mode: str = "sim",
    prom_url: str = "http://localhost:30090",
    namespace: str = "default",
    deployment: str = "podinfo",
    model_path: str | Path = "ppo_autoscaler",
    workload_pattern: str = "diurnal",
    steps: int = 120,
    interval_sec: float = 0.01,
) -> None:
    """Run Experiment 8 (RL vs HPA comparison).

    Parameters
    ----------
    mode : str
        "sim" for real-time simulator, "live" for real Kubernetes cluster.
    prom_url : str
        Prometheus URL (only used in "live" mode).
    namespace : str
        Kubernetes namespace (only used in "live" mode).
    deployment : str
        Target deployment name.
    model_path : str | Path
        Path to RL model.
    workload_pattern : str
        Traffic profile ("diurnal", "flash_crowd", "noisy").
    steps : int
        Number of steps to run.
    interval_sec : float
        Sleep time between steps for simulation mode.
    """
    logger.info(f"=== Starting Experiment 8: RL vs HPA ({mode.upper()} mode) ===")
    logger.info(f"Workload pattern: {workload_pattern}, Steps: {steps}")

    rl_run_name = f"exp8_rl_{workload_pattern}"
    hpa_run_name = f"exp8_hpa_{workload_pattern}"

    if mode == "sim":
        # 1. Run RL Agent in Simulator
        logger.info("--- Running RL Agent in Simulation ---")
        env_rl = K8sSimEnv(workload_pattern=workload_pattern, seed=42)
        env_rl.reset()
        exec_rl = SimulatedExecutor(env_rl)
        obs_rl = SimulatedObserver(env_rl, exec_rl)
        agent_rl = LiveClusterAgent(
            model_path=model_path,
            run_name=rl_run_name,
            observer=obs_rl,
            executor=exec_rl,
        )
        # Fast-forward simulation sleep patching
        import time
        original_sleep = time.sleep
        def mock_sleep_rl(seconds: float) -> None:
            if seconds > 10.0:
                original_sleep(interval_sec)
            else:
                original_sleep(seconds)
        time.sleep = mock_sleep_rl
        try:
            agent_rl.run_loop(duration_steps=steps, interval_sec=30)
        finally:
            time.sleep = original_sleep

        # 2. Run HPA Baseline in Simulator
        logger.info("--- Running HPA Baseline in Simulation ---")
        env_hpa = K8sSimEnv(workload_pattern=workload_pattern, seed=42)
        env_hpa.reset()
        exec_hpa = SimulatedExecutor(env_hpa)
        obs_hpa = SimulatedObserver(env_hpa, exec_hpa)
        agent_hpa = LiveClusterAgent(
            model_path="nonexistent_model_to_force_hpa",
            run_name=hpa_run_name,
            observer=obs_hpa,
            executor=exec_hpa,
        )
        def mock_sleep_hpa(seconds: float) -> None:
            if seconds > 10.0:
                original_sleep(interval_sec)
            else:
                original_sleep(seconds)
        time.sleep = mock_sleep_hpa
        try:
            agent_hpa.run_loop(duration_steps=steps, interval_sec=30)
        finally:
            time.sleep = original_sleep

    elif mode == "live":
        logger.info("--- Running RL Agent on Live Kubernetes Cluster ---")
        agent_rl = LiveClusterAgent(
            prom_url=prom_url,
            namespace=namespace,
            deployment=deployment,
            model_path=model_path,
            run_name=rl_run_name,
        )
        logger.warning("Please ensure HPA is DISABLED in the cluster before running RL Agent!")
        agent_rl.run_loop(duration_steps=steps)

        logger.info("--- Running HPA Baseline on Live Kubernetes Cluster ---")
        logger.warning("Please ensure HPA is ENABLED (and RL Agent is stopped) in the cluster before running HPA!")
        agent_hpa = LiveClusterAgent(
            prom_url=prom_url,
            namespace=namespace,
            deployment=deployment,
            model_path="nonexistent_model_to_force_hpa",
            run_name=hpa_run_name,
        )
        agent_hpa.run_loop(duration_steps=steps)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Analyze and Compare Results
    analyze_results(rl_run_name, hpa_run_name)


def analyze_results(rl_run_name: str, hpa_run_name: str) -> None:
    # Find CSV paths (MetricsLogger saves to logs/live/<prefix>_<timestamp>.csv)
    csv_dir = Path("logs/live")
    if not csv_dir.exists():
        logger.error("No metrics logs found in logs/live")
        return

    rl_files = sorted(csv_dir.glob(f"{rl_run_name}_*.csv"))
    hpa_files = sorted(csv_dir.glob(f"{hpa_run_name}_*.csv"))

    if not rl_files or not hpa_files:
        logger.error("Could not find CSV files for both runs.")
        return

    rl_csv = rl_files[-1]
    hpa_csv = hpa_files[-1]

    df_rl = pd.read_csv(rl_csv)
    df_hpa = pd.read_csv(hpa_csv)

    # Compute metrics
    sla_rl = compute_sla_compliance(df_rl["p99_latency"].tolist())
    sla_hpa = compute_sla_compliance(df_hpa["p99_latency"].tolist())

    cost_rl = compute_average_cost(df_rl["cost_rate"].tolist())
    cost_hpa = compute_average_cost(df_hpa["cost_rate"].tolist())

    churn_rl = compute_churn(df_rl["replicas"].tolist())
    churn_hpa = compute_churn(df_hpa["replicas"].tolist())

    max_lat_rl = compute_max_latency(df_rl["p99_latency"].tolist())
    max_lat_hpa = compute_max_latency(df_hpa["p99_latency"].tolist())

    print("\n" + "="*60)
    print("      EXPERIMENT 8 COMPARISON: RL vs HPA")
    print("="*60)
    print("Metric          | RL Agent (+Safety) | HPA Baseline")
    print("----------------+--------------------+-------------")
    print(f"SLA Compliance  | {sla_rl:17.2f}% | {sla_hpa:11.2f}%")
    print(f"Average Cost    | ${cost_rl:15.2f}/hr | ${cost_hpa:9.2f}/hr")
    print(f"Replica Churn   | {churn_rl:18d} | {churn_hpa:11d}")
    print(f"Max P99 Latency | {max_lat_rl:14.2f}ms | {max_lat_hpa:9.2f}ms")
    print("="*60)

    # Plot runs
    plot_live_run(rl_csv)
    plot_live_run(hpa_csv)
    logger.info("Trajectory plots generated in ./plots/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 8 (RL vs HPA comparison).")
    parser.add_argument("--mode", choices=["sim", "live"], default="sim", help="Execution mode.")
    parser.add_argument("--prom", default="http://localhost:30090", help="Prometheus URL (live mode).")
    parser.add_argument("--namespace", default="default", help="K8s namespace (live mode).")
    parser.add_argument("--deployment", default="podinfo", help="Target deployment (live mode).")
    parser.add_argument("--model", default="ppo_autoscaler", help="Path to RL model.")
    parser.add_argument("--workload", default="diurnal", help="Workload pattern (sim mode).")
    parser.add_argument("--steps", type=int, default=120, help="Steps to run.")
    parser.add_argument("--interval", type=float, default=0.01, help="Sleep interval for fast-forward (sim mode).")
    args = parser.parse_args()

    run_experiment(
        mode=args.mode,
        prom_url=args.prom,
        namespace=args.namespace,
        deployment=args.deployment,
        model_path=args.model,
        workload_pattern=args.workload,
        steps=args.steps,
        interval_sec=args.interval,
    )


if __name__ == "__main__":
    main()
