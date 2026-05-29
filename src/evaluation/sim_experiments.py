"""
Sim Experiments — Runs Experiments 1-7 in the Gymnasium simulator.

Executes predefined protocols to compare RL+Safety vs HPA Baseline,
ablation studies, and structured vs unstructured world models.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import pandas as pd

from src.agents.agent import ContainerScaleAgent
from src.agents.hpa_baseline import RealisticHPA
from src.env.k8s_sim import K8sSimEnv
from src.env.workload import PATTERNS, HELD_OUT_PATTERNS
from src.evaluation.metrics import (
    compute_average_cost,
    compute_churn,
    compute_max_latency,
    compute_sla_compliance,
    compute_average_latency,
    compute_percentile_latency,
    compute_violation_severity,
    compute_mean_utilization,
    compute_total_reward,
)
from src.safety.safety_filter import ClusterState

logger = logging.getLogger(__name__)


def composite_score(sla_compliance: float, avg_cost: float, baseline_cost: float = 0.65) -> float:
    """Single balanced score: higher is better.

    Combines SLA compliance (0-100%) and cost efficiency into one number.
    Useful for quick comparison: if RL score > HPA score, RL wins overall.
    """
    sla_score = sla_compliance / 100.0
    cost_efficiency = max(0.0, 1.0 - max(0.0, avg_cost - baseline_cost) / max(baseline_cost, 0.01))
    return 0.5 * sla_score + 0.5 * cost_efficiency


def run_episode(
    env: K8sSimEnv,
    agent: Any,
    is_hpa: bool = False,
    sla_target: float = 200.0,
) -> dict[str, Any]:
    """Run a single episode and return metrics."""
    obs, info = env.reset()
    agent.reset()

    latencies = []
    costs = []
    replicas = []
    rewards = []
    cpu_utils = []

    terminated = False
    truncated = False
    step = 0

    while not (terminated or truncated):
        if is_hpa:
            state = ClusterState.from_obs(obs, sla_target=sla_target)
            action_delta = agent.act(state)
            action_delta = max(-3, min(3, action_delta))
            action = action_delta + 3
        else:
            delta = agent.decide(obs, step)
            action = delta + 3

        obs, reward, terminated, truncated, info = env.step(action)

        latencies.append(info["p99_latency"])
        costs.append(info["cost_rate"])
        replicas.append(info["replicas"])
        rewards.append(reward)
        cpu_utils.append(info["cpu_util"])
        step += 1

    return {
        "sla_compliance": compute_sla_compliance(latencies, sla_target),
        "avg_cost": compute_average_cost(costs),
        "churn": compute_churn(replicas),
        "max_latency": compute_max_latency(latencies),
        "avg_latency": compute_average_latency(latencies),
        "p99_latency": compute_percentile_latency(latencies, 99.0),
        "severity": compute_violation_severity(latencies, sla_target),
        "cpu_util": compute_mean_utilization(cpu_utils),
        "reward": compute_total_reward(rewards),
        "pattern": info["workload_pattern"],
    }


def _run_paired_episodes(
    n_episodes: int,
    rl_agent: ContainerScaleAgent,
    hpa_agent: RealisticHPA,
    workload_pattern: str = "random",
    base_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run paired RL vs HPA episodes on the same seeds for fair comparison."""
    env = K8sSimEnv(workload_pattern=workload_pattern)
    rl_results = []
    hpa_results = []

    for i in range(n_episodes):
        seed = base_seed + i

        # RL run
        env.reset(seed=seed)
        rl_results.append(run_episode(env, rl_agent, is_hpa=False))

        # HPA run (same seed = same traffic)
        env.reset(seed=seed)
        hpa_results.append(run_episode(env, hpa_agent, is_hpa=True))

    return pd.DataFrame(rl_results), pd.DataFrame(hpa_results)


def _print_comparison(df_rl: pd.DataFrame, df_hpa: pd.DataFrame, title: str) -> None:
    """Print a formatted comparison table of RL vs HPA metrics."""
    rl_score = composite_score(df_rl["sla_compliance"].mean(), df_rl["avg_cost"].mean())
    hpa_score = composite_score(df_hpa["sla_compliance"].mean(), df_hpa["avg_cost"].mean())

    print(f"\n=== {title} ===")
    print(f"{'Metric':<23} | {'RL+Safety':>10} | {'HPA Baseline':>12}")
    print("-" * 23 + "-+-" + "-" * 10 + "-+-" + "-" * 12)
    print(f"{'SLA Compliance':<23} | {df_rl['sla_compliance'].mean():>9.2f}% | {df_hpa['sla_compliance'].mean():>11.2f}%")
    print(f"{'Violation Severity':<23} | {df_rl['severity'].mean():>8.0f}ms | {df_hpa['severity'].mean():>10.0f}ms")
    print(f"{'Average Cost':<23} | ${df_rl['avg_cost'].mean():>8.2f}/hr | ${df_hpa['avg_cost'].mean():>9.2f}/hr")
    print(f"{'Replica Churn':<23} | {df_rl['churn'].mean():>10.1f} | {df_hpa['churn'].mean():>12.1f}")
    print(f"{'Average Latency':<23} | {df_rl['avg_latency'].mean():>8.0f}ms | {df_hpa['avg_latency'].mean():>10.0f}ms")
    print(f"{'P99 Latency':<23} | {df_rl['p99_latency'].mean():>8.0f}ms | {df_hpa['p99_latency'].mean():>10.0f}ms")
    print(f"{'Max Latency':<23} | {df_rl['max_latency'].mean():>8.0f}ms | {df_hpa['max_latency'].mean():>10.0f}ms")
    print(f"{'CPU Utilization':<23} | {df_rl['cpu_util'].mean()*100:>9.1f}% | {df_hpa['cpu_util'].mean()*100:>11.1f}%")
    print(f"{'Total Reward':<23} | {df_rl['reward'].mean():>10.2f} | {df_hpa['reward'].mean():>12.2f}")
    print(f"{'Composite Score':<23} | {rl_score:>10.3f} | {hpa_score:>12.3f}")

    winner = "RL" if rl_score > hpa_score else "HPA"
    print(f"\n  → Winner (composite): {winner}")


def run_exp1_baseline(n_episodes: int = 100, model_path: str = "ppo_autoscaler") -> None:
    """Experiment 1: RL+Safety vs RealisticHPA across all traffic patterns."""
    logger.info("Running Experiment 1: Baseline Comparison using model %s", model_path)

    try:
        rl_agent = ContainerScaleAgent(model_path=model_path)
    except Exception as e:
        logger.error("Cannot run Exp1 without trained RL model: %s", e)
        return

    hpa_agent = RealisticHPA()

    # Overall comparison (random patterns)
    df_rl, df_hpa = _run_paired_episodes(n_episodes, rl_agent, hpa_agent)
    _print_comparison(df_rl, df_hpa, f"Experiment 1: Overall ({n_episodes} episodes)")

    # Per-pattern breakdown
    for pattern in PATTERNS:
        df_rl_p = df_rl[df_rl["pattern"] == pattern]
        df_hpa_p = df_hpa[df_hpa["pattern"] == pattern]
        if len(df_rl_p) > 0:
            _print_comparison(df_rl_p, df_hpa_p, f"  Pattern: {pattern}")


def run_exp9_generalization(n_episodes: int = 50, model_path: str = "ppo_autoscaler") -> None:
    """Experiment 9: Test agent on held-out workload patterns it never trained on.

    Compares performance on training patterns vs held-out patterns.
    If held-out performance degrades >10%, the agent has overfit.
    """
    logger.info("Running Experiment 9: Generalization Test using model %s", model_path)

    try:
        rl_agent = ContainerScaleAgent(model_path=model_path)
    except Exception as e:
        logger.error("Cannot run Exp9 without trained RL model: %s", e)
        return

    hpa_agent = RealisticHPA()

    # Run on training patterns
    print("\n--- Training Patterns ---")
    train_rl_results = []
    train_hpa_results = []
    for pattern in PATTERNS:
        env = K8sSimEnv(workload_pattern=pattern)
        for i in range(n_episodes // len(PATTERNS)):
            seed = 1000 + i
            env.reset(seed=seed)
            train_rl_results.append(run_episode(env, rl_agent, is_hpa=False))
            env.reset(seed=seed)
            train_hpa_results.append(run_episode(env, hpa_agent, is_hpa=True))

    df_train_rl = pd.DataFrame(train_rl_results)
    df_train_hpa = pd.DataFrame(train_hpa_results)
    _print_comparison(df_train_rl, df_train_hpa, "Training Patterns (seen during training)")

    # Run on held-out patterns
    print("\n--- Held-Out Patterns ---")
    held_rl_results = []
    held_hpa_results = []
    for pattern in HELD_OUT_PATTERNS:
        env = K8sSimEnv(workload_pattern=pattern)
        for i in range(n_episodes // len(HELD_OUT_PATTERNS)):
            seed = 2000 + i
            env.reset(seed=seed)
            held_rl_results.append(run_episode(env, rl_agent, is_hpa=False))
            env.reset(seed=seed)
            held_hpa_results.append(run_episode(env, hpa_agent, is_hpa=True))

    df_held_rl = pd.DataFrame(held_rl_results)
    df_held_hpa = pd.DataFrame(held_hpa_results)
    _print_comparison(df_held_rl, df_held_hpa, "Held-Out Patterns (never seen during training)")

    # Generalization gap
    train_score = composite_score(
        df_train_rl["sla_compliance"].mean(), df_train_rl["avg_cost"].mean()
    )
    held_score = composite_score(
        df_held_rl["sla_compliance"].mean(), df_held_rl["avg_cost"].mean()
    )
    gap = abs(train_score - held_score) / max(train_score, 0.01) * 100

    print(f"\n=== Generalization Gap ===")
    print(f"  Training composite score:  {train_score:.3f}")
    print(f"  Held-out composite score:  {held_score:.3f}")
    print(f"  Gap: {gap:.1f}%")
    if gap > 10:
        print("  ⚠️  Gap > 10% — possible overfitting detected!")
    else:
        print("  ✓  Gap ≤ 10% — agent generalizes well.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, default=1, help="Experiment number (1 or 9)")
    parser.add_argument("--model", type=str, default="ppo_autoscaler",
                        help="Path to the RL model (default: ppo_autoscaler)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes to run (default: 100)")
    args = parser.parse_args()

    if args.exp == 1:
        run_exp1_baseline(n_episodes=args.episodes, model_path=args.model)
    elif args.exp == 9:
        run_exp9_generalization(n_episodes=args.episodes, model_path=args.model)
    else:
        logger.info("Experiment %d is implemented as a stub.", args.exp)


if __name__ == "__main__":
    main()
