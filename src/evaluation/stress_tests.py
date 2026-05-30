"""
Stress Tests — Probes agent robustness under extreme conditions.

Five targeted experiments:
  A. Safety filter ablation
  B. Adversarial workload stress
  C. Sensitivity analysis (parameter perturbation)
  D. Multi-seed stability
  E. Scalability boundary

Usage:
    python -m src.evaluation.stress_tests --model ppo_autoscaler
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents.agent import ContainerScaleAgent
from src.agents.hpa_baseline import RealisticHPA
from src.env.k8s_sim import K8sSimEnv
from src.env.workload import PATTERNS
from src.evaluation.extended_metrics import (
    compute_coefficient_of_variation, bootstrap_ci,
)
from src.evaluation.metrics import (
    compute_average_cost, compute_average_latency, compute_churn,
    compute_max_latency, compute_sla_compliance, compute_total_reward,
    compute_violation_severity,
)
from src.evaluation.sim_experiments import composite_score
from src.safety.safety_filter import ClusterState, SafetyFilter

logger = logging.getLogger(__name__)


# ======================================================================
# Helpers
# ======================================================================

class NoOpSafetyFilter:
    """Passes all deltas through unchanged — for ablation testing."""
    def check(self, state, proposed_delta, step):
        return proposed_delta
    def reset(self):
        pass


def _run_one_episode(env, agent, agent_type, sla_target=200.0):
    """Minimal episode runner returning core metrics."""
    obs, info = env.reset()
    if hasattr(agent, "reset"):
        agent.reset()

    lats, costs, reps, rews, cpus = [], [], [], [], []
    terminated = truncated = False
    step = 0

    while not (terminated or truncated):
        if agent_type == "rl":
            delta = agent.decide(obs, step)
            action = delta + 3
        else:
            state = ClusterState.from_obs(obs, sla_target=sla_target)
            action_delta = max(-3, min(3, agent.act(state)))
            action = action_delta + 3

        obs, reward, terminated, truncated, info = env.step(action)
        lats.append(info["p99_latency"])
        costs.append(info["cost_rate"])
        reps.append(info["replicas"])
        rews.append(reward)
        cpus.append(info["cpu_util"])
        step += 1

    return {
        "sla_compliance": compute_sla_compliance(lats, sla_target),
        "avg_cost": compute_average_cost(costs),
        "churn": compute_churn(reps),
        "max_latency": compute_max_latency(lats),
        "avg_latency": compute_average_latency(lats),
        "severity": compute_violation_severity(lats, sla_target),
        "reward": compute_total_reward(rews),
        "composite": composite_score(
            compute_sla_compliance(lats, sla_target),
            compute_average_cost(costs),
        ),
        "pattern": info.get("workload_pattern", "unknown"),
    }


# ======================================================================
# Experiment A: Safety Filter Ablation
# ======================================================================

def run_exp_a_ablation(model_path: str, n_episodes: int = 50) -> dict:
    """Compare RL+SafetyFilter vs RL-raw (no filter)."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT A: Safety Filter Ablation")
    print("=" * 70)

    try:
        agent_safe = ContainerScaleAgent(model_path=model_path)
        agent_raw = ContainerScaleAgent(model_path=model_path)
    except Exception as e:
        print(f"  ⚠ Cannot load model: {e}")
        return {}

    # Disable safety filter on the raw agent
    agent_raw.safety = NoOpSafetyFilter()

    safe_results, raw_results = [], []

    for i in range(n_episodes):
        seed = 5000 + i
        env = K8sSimEnv(workload_pattern="random")

        env.reset(seed=seed)
        safe_results.append(_run_one_episode(env, agent_safe, "rl"))

        env.reset(seed=seed)
        raw_results.append(_run_one_episode(env, agent_raw, "rl"))

    df_safe = pd.DataFrame(safe_results)
    df_raw = pd.DataFrame(raw_results)

    metrics = ["sla_compliance", "avg_cost", "max_latency", "churn", "composite"]
    print(f"\n  {'Metric':<20} | {'With Safety':<20} | {'Without Safety':<20}")
    print("  " + "-" * 65)
    for m in metrics:
        s_mean, s_std = df_safe[m].mean(), df_safe[m].std()
        r_mean, r_std = df_raw[m].mean(), df_raw[m].std()
        print(f"  {m:<20} | {s_mean:>8.2f} ± {s_std:>6.2f} | {r_mean:>8.2f} ± {r_std:>6.2f}")

    # Crash analysis (max_latency > 5x SLA)
    crash_safe = (df_safe["max_latency"] > 1000).sum()
    crash_raw = (df_raw["max_latency"] > 1000).sum()
    print(f"\n  Crash episodes (max_lat > 1000ms):")
    print(f"    With safety:    {crash_safe}/{n_episodes} ({crash_safe/n_episodes*100:.1f}%)")
    print(f"    Without safety: {crash_raw}/{n_episodes} ({crash_raw/n_episodes*100:.1f}%)")

    return {"safe": df_safe.to_dict("records"), "raw": df_raw.to_dict("records")}


# ======================================================================
# Experiment B: Adversarial Workloads
# ======================================================================

def _adversarial_step(env: K8sSimEnv, pattern: str, step: int) -> None:
    """Override the environment's request rate with adversarial patterns."""
    if pattern == "instant_spike":
        env.request_rate = 50.0 if step < 30 else 500.0
    elif pattern == "oscillating_extreme":
        env.request_rate = 50.0 if step % 6 < 3 else 450.0
    elif pattern == "sustained_overload":
        env.request_rate = 600.0
    elif pattern == "dead_then_spike":
        env.request_rate = 5.0 if step < 60 else 400.0


def run_exp_b_adversarial(model_path: str, n_episodes: int = 20) -> dict:
    """Test RL agent under extreme adversarial workloads."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT B: Adversarial Workload Stress Test")
    print("=" * 70)

    try:
        rl_agent = ContainerScaleAgent(model_path=model_path)
    except Exception as e:
        print(f"  ⚠ Cannot load model: {e}")
        return {}

    hpa_agent = RealisticHPA()
    adv_patterns = ["instant_spike", "oscillating_extreme", "sustained_overload", "dead_then_spike"]
    all_results = {}

    for adv_pat in adv_patterns:
        rl_res, hpa_res = [], []
        for i in range(n_episodes):
            seed = 6000 + i

            for agent, agent_type, result_list in [
                (rl_agent, "rl", rl_res), (hpa_agent, "hpa", hpa_res)
            ]:
                env = K8sSimEnv(workload_pattern="steady")
                obs, info = env.reset(seed=seed)
                if hasattr(agent, "reset"):
                    agent.reset()

                lats, costs, reps, rews = [], [], [], []
                terminated = truncated = False
                step = 0

                while not (terminated or truncated):
                    _adversarial_step(env, adv_pat, step)

                    if agent_type == "rl":
                        delta = agent.decide(obs, step)
                        action = delta + 3
                    else:
                        state = ClusterState.from_obs(obs, sla_target=200.0)
                        action = max(-3, min(3, agent.act(state))) + 3

                    obs, reward, terminated, truncated, info = env.step(action)
                    lats.append(info["p99_latency"])
                    costs.append(info["cost_rate"])
                    reps.append(info["replicas"])
                    rews.append(reward)
                    step += 1

                result_list.append({
                    "sla_compliance": compute_sla_compliance(lats),
                    "avg_cost": compute_average_cost(costs),
                    "max_latency": compute_max_latency(lats),
                    "churn": compute_churn(reps),
                    "reward": compute_total_reward(rews),
                })

        df_rl = pd.DataFrame(rl_res)
        df_hpa = pd.DataFrame(hpa_res)

        print(f"\n  ── {adv_pat} ──")
        print(f"  {'Metric':<18} | {'RL':<18} | {'HPA':<18}")
        print("  " + "-" * 58)
        for m in ["sla_compliance", "avg_cost", "max_latency", "churn"]:
            print(f"  {m:<18} | {df_rl[m].mean():>7.2f} ± {df_rl[m].std():>5.2f}"
                  f" | {df_hpa[m].mean():>7.2f} ± {df_hpa[m].std():>5.2f}")

        all_results[adv_pat] = {
            "rl": df_rl.to_dict("records"), "hpa": df_hpa.to_dict("records")
        }

    return all_results


# ======================================================================
# Experiment C: Sensitivity Analysis
# ======================================================================

def run_exp_c_sensitivity(model_path: str, n_episodes: int = 30) -> dict:
    """Perturb simulator params ±20% and measure performance shift."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT C: Sensitivity Analysis (±20% perturbation)")
    print("=" * 70)

    try:
        agent = ContainerScaleAgent(model_path=model_path)
    except Exception as e:
        print(f"  ⚠ Cannot load model: {e}")
        return {}

    # Baseline run
    baseline_scores = []
    for i in range(n_episodes):
        env = K8sSimEnv(workload_pattern="random")
        env.reset(seed=7000 + i)
        r = _run_one_episode(env, agent, "rl")
        baseline_scores.append(r["composite"])
    baseline_mean = np.mean(baseline_scores)

    # Parameters to perturb
    params = {
        "per_pod_capacity": ("_ppc_range", (15, 30)),
        "cold_start_mean": ("_csm_range", (40, 100)),
        "node_price": ("_np_range", (0.20, 0.50)),
        "sla_target": ("sla_target", 200),
    }

    sensitivities = {}
    print(f"\n  Baseline composite: {baseline_mean:.4f}")
    print(f"  {'Parameter':<22} | {'−20%':>10} | {'Baseline':>10} | {'+20%':>10} | {'Sensitivity':>12}")
    print("  " + "-" * 75)

    for param_name, (attr, default_val) in params.items():
        for direction, factor in [("-20%", 0.8), ("+20%", 1.2)]:
            scores = []
            for i in range(n_episodes):
                env = K8sSimEnv(workload_pattern="random")

                # Apply perturbation
                if isinstance(default_val, tuple):
                    lo, hi = default_val
                    setattr(env, attr, (lo * factor, hi * factor))
                else:
                    setattr(env, attr, default_val * factor)

                env.reset(seed=7000 + i)
                r = _run_one_episode(env, agent, "rl")
                scores.append(r["composite"])

            key = f"{param_name}_{direction}"
            sensitivities[key] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "delta_pct": float((np.mean(scores) - baseline_mean) / max(baseline_mean, 1e-6) * 100),
            }

        lo_key = f"{param_name}_-20%"
        hi_key = f"{param_name}_+20%"
        lo_val = sensitivities[lo_key]["mean"]
        hi_val = sensitivities[hi_key]["mean"]
        span = abs(hi_val - lo_val) / max(baseline_mean, 1e-6) * 100
        print(f"  {param_name:<22} | {lo_val:>10.4f} | {baseline_mean:>10.4f} | "
              f"{hi_val:>10.4f} | {span:>10.1f}%")

    return {"baseline": baseline_mean, "sensitivities": sensitivities}


# ======================================================================
# Experiment D: Multi-Seed Stability
# ======================================================================

def run_exp_d_stability(model_path: str, n_seeds: int = 20) -> dict:
    """Run agent with many seeds, compute CV for key metrics."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT D: Multi-Seed Stability Analysis")
    print("=" * 70)

    try:
        agent = ContainerScaleAgent(model_path=model_path)
    except Exception as e:
        print(f"  ⚠ Cannot load model: {e}")
        return {}

    results = []
    for i in range(n_seeds):
        env = K8sSimEnv(workload_pattern="random")
        env.reset(seed=8000 + i * 137)  # spread seeds
        results.append(_run_one_episode(env, agent, "rl"))

    df = pd.DataFrame(results)
    metrics = ["sla_compliance", "avg_cost", "max_latency", "reward", "composite"]

    print(f"\n  {'Metric':<20} | {'Mean':>10} | {'Std':>10} | {'CV':>8} | {'Status':<12}")
    print("  " + "-" * 70)
    stability_report = {}
    for m in metrics:
        vals = df[m].tolist()
        cv = compute_coefficient_of_variation(vals)
        _, ci_lo, ci_hi = bootstrap_ci(vals)
        status = "⚠ UNSTABLE" if cv > 0.15 else "✓ stable"
        print(f"  {m:<20} | {np.mean(vals):>10.3f} | {np.std(vals):>10.3f} | "
              f"{cv:>7.3f} | {status}")
        stability_report[m] = {"cv": cv, "ci95": [ci_lo, ci_hi], "status": status}

    return stability_report


# ======================================================================
# Experiment E: Scalability Boundary
# ======================================================================

def run_exp_e_scalability(model_path: str, n_episodes: int = 10) -> dict:
    """Test with increasing request rates to find breaking point."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT E: Scalability Boundary Test")
    print("=" * 70)

    try:
        agent = ContainerScaleAgent(model_path=model_path)
    except Exception as e:
        print(f"  ⚠ Cannot load model: {e}")
        return {}

    rate_multipliers = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    results = {}

    print(f"\n  {'Multiplier':>12} | {'SLA%':>8} | {'Cost':>8} | {'MaxLat':>10} | {'Composite':>10}")
    print("  " + "-" * 60)

    for mult in rate_multipliers:
        ep_results = []
        for i in range(n_episodes):
            env = K8sSimEnv(workload_pattern="random")
            obs, info = env.reset(seed=9000 + i)
            agent.reset()

            lats, costs, reps, rews = [], [], [], []
            terminated = truncated = False
            step = 0

            while not (terminated or truncated):
                # Amplify request rate
                env.request_rate *= mult

                delta = agent.decide(obs, step)
                action = delta + 3
                obs, reward, terminated, truncated, info = env.step(action)
                lats.append(info["p99_latency"])
                costs.append(info["cost_rate"])
                reps.append(info["replicas"])
                rews.append(reward)
                step += 1

            ep_results.append({
                "sla_compliance": compute_sla_compliance(lats),
                "avg_cost": compute_average_cost(costs),
                "max_latency": compute_max_latency(lats),
                "composite": composite_score(
                    compute_sla_compliance(lats), compute_average_cost(costs)),
            })

        df = pd.DataFrame(ep_results)
        results[f"{mult}x"] = df.to_dict("records")
        print(f"  {mult:>10.1f}x  | {df['sla_compliance'].mean():>7.1f}% | "
              f"${df['avg_cost'].mean():>6.3f} | "
              f"{df['max_latency'].mean():>8.1f}ms | "
              f"{df['composite'].mean():>10.4f}")

    return results


# ======================================================================
# Main
# ======================================================================

def run_all_stress_tests(model_path: str, output_dir: str = "results/stress_tests") -> None:
    """Run all 5 stress test experiments."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_results["exp_a_ablation"] = run_exp_a_ablation(model_path)
    all_results["exp_b_adversarial"] = run_exp_b_adversarial(model_path)
    all_results["exp_c_sensitivity"] = run_exp_c_sensitivity(model_path)
    all_results["exp_d_stability"] = run_exp_d_stability(model_path)
    all_results["exp_e_scalability"] = run_exp_e_scalability(model_path)

    (out / "stress_test_results.json").write_text(
        json.dumps(all_results, indent=2, default=str))
    print(f"\n  Results saved to {out}/stress_test_results.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="RL Agent Stress Tests")
    parser.add_argument("--model", default="ppo_autoscaler", help="Model path")
    parser.add_argument("--output-dir", default="results/stress_tests")
    parser.add_argument("--exp", type=str, default="all",
                        choices=["all", "a", "b", "c", "d", "e"],
                        help="Which experiment to run (default: all)")
    args = parser.parse_args()

    if args.exp == "all":
        run_all_stress_tests(args.model, args.output_dir)
    elif args.exp == "a":
        run_exp_a_ablation(args.model)
    elif args.exp == "b":
        run_exp_b_adversarial(args.model)
    elif args.exp == "c":
        run_exp_c_sensitivity(args.model)
    elif args.exp == "d":
        run_exp_d_stability(args.model)
    elif args.exp == "e":
        run_exp_e_scalability(args.model)


if __name__ == "__main__":
    main()
