"""
Benchmark Runner — Multi-agent, multi-pattern, multi-seed evaluation harness.

Usage:
    python -m src.evaluation.benchmark_runner \
        --agents ppo_autoscaler dqn_autoscaler qrdqn_autoscaler \
        --patterns all --seeds 5 --episodes-per-seed 20
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents.agent import ContainerScaleAgent
from src.agents.hpa_baseline import RealisticHPA
from src.env.k8s_sim import K8sSimEnv
from src.env.workload import PATTERNS, HELD_OUT_PATTERNS, ALL_PATTERNS, WorkloadGenerator
from src.evaluation.baselines import FixedReplicaBaseline, OracleScaler, ReactiveThresholdBaseline
from src.evaluation.extended_metrics import (
    bootstrap_ci, bonferroni_correct, compute_action_entropy,
    compute_coefficient_of_variation, compute_jain_fairness,
    compute_over_provisioning_ratio, compute_reaction_time,
    compute_settling_time, compute_under_provisioning_ratio,
    mann_whitney_with_effect_size,
)
from src.evaluation.metrics import (
    compute_average_cost, compute_average_latency, compute_churn,
    compute_max_latency, compute_mean_utilization, compute_percentile_latency,
    compute_sla_compliance, compute_total_reward, compute_violation_severity,
)
from src.evaluation.sim_experiments import composite_score
from src.safety.safety_filter import ClusterState

logger = logging.getLogger(__name__)


def run_episode_extended(
    env: K8sSimEnv, agent: Any, agent_type: str, sla_target: float = 200.0,
) -> dict[str, Any]:
    """Run one episode collecting base + extended metrics."""
    obs, info = env.reset()
    if hasattr(agent, "reset"):
        agent.reset()

    latencies, costs, replicas_h = [], [], []
    rewards, cpu_utils, request_rates, actions = [], [], [], []
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
        latencies.append(info["p99_latency"])
        costs.append(info["cost_rate"])
        replicas_h.append(info["replicas"])
        rewards.append(reward)
        cpu_utils.append(info["cpu_util"])
        request_rates.append(info["request_rate"])
        actions.append(action)
        step += 1

    result = {
        "sla_compliance": compute_sla_compliance(latencies, sla_target),
        "avg_cost": compute_average_cost(costs),
        "churn": compute_churn(replicas_h),
        "max_latency": compute_max_latency(latencies),
        "avg_latency": compute_average_latency(latencies),
        "p95_latency": compute_percentile_latency(latencies, 95.0),
        "p99_latency": compute_percentile_latency(latencies, 99.0),
        "severity": compute_violation_severity(latencies, sla_target),
        "cpu_util": compute_mean_utilization(cpu_utils),
        "reward": compute_total_reward(rewards),
        "pattern": info["workload_pattern"],
        "reaction_time": compute_reaction_time(request_rates, replicas_h),
        "over_provisioning": compute_over_provisioning_ratio(
            request_rates, replicas_h, env.per_pod_capacity),
        "under_provisioning": compute_under_provisioning_ratio(cpu_utils),
        "settling_time": compute_settling_time(latencies, sla_target),
        "action_entropy": compute_action_entropy(actions),
        "replica_fairness": compute_jain_fairness(replicas_h),
    }
    result["composite"] = composite_score(result["sla_compliance"], result["avg_cost"])
    result["_trajectories"] = {
        "latencies": latencies, "costs": costs, "replicas": replicas_h,
        "cpu_utils": cpu_utils, "request_rates": request_rates,
        "actions": actions, "rewards": rewards,
    }
    return result


def _load_agents(model_paths: list[str]) -> dict[str, tuple[Any, str]]:
    """Load all agents. RL agents that fail are skipped with a warning."""
    agents: dict[str, tuple[Any, str]] = {}
    agents["HPA"] = (RealisticHPA(), "hpa")
    agents["Fixed-5"] = (FixedReplicaBaseline(fixed_replicas=5), "fixed")
    agents["Fixed-10"] = (FixedReplicaBaseline(fixed_replicas=10), "fixed")
    agents["Reactive"] = (ReactiveThresholdBaseline(), "reactive")

    for path in model_paths:
        # Special case: ensemble meta-agent
        if path == "ensemble":
            try:
                from src.agents.ensemble_agent import EnsembleMetaAgent
                agents["RL-Ensemble"] = (EnsembleMetaAgent(), "rl")
                logger.info("Loaded Ensemble Meta-Agent")
            except Exception as e:
                logger.warning("Skipping ensemble — could not load: %s", e)
            continue

        name = Path(path).stem.replace("_autoscaler", "").upper()
        try:
            agents[f"RL-{name}"] = (ContainerScaleAgent(model_path=path), "rl")
            logger.info("Loaded RL agent: %s from %s", name, path)
        except Exception as e:
            logger.warning("Skipping %s — could not load: %s", path, e)

    return agents


def run_benchmark(
    model_paths: list[str], patterns: list[str],
    n_seeds: int = 5, episodes_per_seed: int = 20,
    output_dir: str = "results/benchmark", sla_target: float = 200.0,
) -> Path:
    """Run the full benchmark and save results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots" / "trajectory_overlays").mkdir(parents=True, exist_ok=True)

    run_config = {
        "model_paths": model_paths, "patterns": patterns,
        "n_seeds": n_seeds, "episodes_per_seed": episodes_per_seed,
        "sla_target": sla_target, "timestamp": datetime.now().isoformat(),
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2))

    agents = _load_agents(model_paths)
    logger.info("Loaded %d agents: %s", len(agents), list(agents.keys()))

    total = len(agents) * len(patterns) * n_seeds * episodes_per_seed
    logger.info("Running %d total episodes...", total)

    try:
        from tqdm import tqdm
        progress = tqdm(total=total, desc="Benchmark", unit="ep")
    except ImportError:
        progress = None

    all_results: list[dict[str, Any]] = []
    trajectory_store: dict[str, dict] = {}
    start_time = time.time()

    for agent_name, (agent, agent_type) in agents.items():
        for pattern in patterns:
            for seed_idx in range(n_seeds):
                base_seed = 1000 * (seed_idx + 1)
                for ep_idx in range(episodes_per_seed):
                    seed = base_seed + ep_idx

                    if agent_name.startswith("Oracle"):
                        wg = WorkloadGenerator(pattern=pattern, seed=seed)
                        rates = [wg.get_rate(s) for s in range(120)]
                        cur_agent = OracleScaler(future_rates=rates)
                        cur_type = "oracle"
                    else:
                        cur_agent, cur_type = agent, agent_type

                    env = K8sSimEnv(workload_pattern=pattern)
                    env.reset(seed=seed)
                    result = run_episode_extended(env, cur_agent, cur_type, sla_target)

                    result["agent"] = agent_name
                    result["seed"] = seed
                    result["seed_idx"] = seed_idx
                    result["episode_idx"] = ep_idx

                    tkey = f"{agent_name}_{pattern}"
                    if tkey not in trajectory_store and seed_idx == 0 and ep_idx == 0:
                        trajectory_store[tkey] = {
                            "agent": agent_name, "pattern": pattern,
                            **result["_trajectories"],
                        }
                    del result["_trajectories"]
                    all_results.append(result)
                    if progress:
                        progress.update(1)

    if progress:
        progress.close()

    elapsed = time.time() - start_time
    logger.info("Completed in %.1fs (%.0f ep/s)", elapsed, total / max(elapsed, 0.01))

    df = pd.DataFrame(all_results)
    df.to_csv(out / "raw_results.csv", index=False)
    (out / "raw_results.json").write_text(
        json.dumps(all_results, indent=2, default=str))
    (out / "trajectories.json").write_text(
        json.dumps(trajectory_store, default=lambda o: int(o) if isinstance(o, np.integer) else float(o)))

    summary = _build_summary_table(df)
    summary.to_csv(out / "summary_table.csv")

    stat_results = _run_statistical_tests(df)
    (out / "statistical_tests.json").write_text(
        json.dumps(stat_results, indent=2, default=str))

    _print_summary(df, stat_results, elapsed)

    try:
        from src.evaluation.plot_results import generate_all_plots
        generate_all_plots(out)
    except Exception as e:
        logger.warning("Plot generation failed: %s", e)

    return out


def _build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "sla_compliance", "avg_cost", "churn", "avg_latency", "p99_latency",
        "max_latency", "severity", "cpu_util", "reward", "composite",
        "reaction_time", "over_provisioning", "under_provisioning",
        "settling_time", "action_entropy", "replica_fairness",
    ]
    rows = []
    for agent in df["agent"].unique():
        for pattern in sorted(df["pattern"].unique()):
            subset = df[(df["agent"] == agent) & (df["pattern"] == pattern)]
            if len(subset) == 0:
                continue
            row: dict[str, Any] = {"agent": agent, "pattern": pattern, "n": len(subset)}
            for m in metrics:
                if m not in subset.columns:
                    continue
                vals = subset[m].dropna()
                row[f"{m}_mean"] = vals.mean()
                row[f"{m}_std"] = vals.std() if len(vals) > 1 else 0.0
                if len(vals) > 1:
                    _, lo, hi = bootstrap_ci(vals.tolist(), n_bootstrap=2000)
                    row[f"{m}_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
            rows.append(row)
    return pd.DataFrame(rows)


def _run_statistical_tests(df: pd.DataFrame) -> dict[str, Any]:
    agent_names = sorted(df["agent"].unique())
    test_metrics = ["composite", "sla_compliance", "avg_cost", "reward"]
    tests, raw_p = [], []

    for i, a1 in enumerate(agent_names):
        for a2 in agent_names[i + 1:]:
            for m in test_metrics:
                va = df[df["agent"] == a1][m].dropna().tolist()
                vb = df[df["agent"] == a2][m].dropna().tolist()
                if len(va) < 2 or len(vb) < 2:
                    continue
                r = mann_whitney_with_effect_size(va, vb)
                r.update({"agent_a": a1, "agent_b": a2, "metric": m})
                tests.append(r)
                raw_p.append(r["p_value"])

    corrected = bonferroni_correct(raw_p)
    for t, pc in zip(tests, corrected):
        t["p_corrected"] = pc
        t["sig_corrected"] = pc < 0.05

    stability = {}
    for an in agent_names:
        ad = df[df["agent"] == an]
        stability[an] = {
            m: compute_coefficient_of_variation(ad[m].dropna().tolist())
            for m in test_metrics if m in ad.columns
        }

    return {"pairwise": tests, "stability": stability}


def _print_summary(df: pd.DataFrame, stats: dict, elapsed: float) -> None:
    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"  Episodes: {len(df)} | Time: {elapsed:.1f}s | "
          f"Agents: {', '.join(sorted(df['agent'].unique()))}")

    print("\n  ── OVERALL RANKING (composite) ──")
    rank = df.groupby("agent")["composite"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    for i, (ag, r) in enumerate(rank.iterrows(), 1):
        print(f"    #{i}  {ag:<15}  {r['mean']:.4f} ± {r['std']:.4f}")

    print("\n  ── SLA COMPLIANCE ──")
    sr = df.groupby("agent")["sla_compliance"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    for i, (ag, r) in enumerate(sr.iterrows(), 1):
        print(f"    #{i}  {ag:<15}  {r['mean']:.1f}% ± {r['std']:.1f}%")

    print("\n  ── COST (lower=better) ──")
    cr = df.groupby("agent")["avg_cost"].agg(["mean", "std"]).sort_values("mean")
    for i, (ag, r) in enumerate(cr.iterrows(), 1):
        print(f"    #{i}  {ag:<15}  ${r['mean']:.3f}/hr ± ${r['std']:.3f}")

    sig = [t for t in stats.get("pairwise", []) if t.get("sig_corrected")]
    if sig:
        print(f"\n  ── SIGNIFICANT ({len(sig)} Bonferroni-corrected) ──")
        for t in sig[:10]:
            d = ">" if t["cohens_d"] > 0 else "<"
            print(f"    {t['agent_a']} {d} {t['agent_b']} [{t['metric']}] "
                  f"d={t['cohens_d']:.2f} ({t['effect_interpretation']}) p={t['p_corrected']:.4f}")

    print("\n  ── STABILITY ──")
    for ag, s in stats.get("stability", {}).items():
        bad = {k: v for k, v in s.items() if v > 0.15}
        if bad:
            print(f"    ⚠  {ag}: high CV in {list(bad.keys())}")
        else:
            print(f"    ✓  {ag}: stable")
    print("=" * 80)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Multi-agent benchmark runner.")
    parser.add_argument("--agents", nargs="+", default=["ppo_autoscaler"])
    parser.add_argument("--patterns", nargs="+", default=["all"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.patterns == ["all"]:
        patterns = list(ALL_PATTERNS)
    elif args.patterns == ["training"]:
        patterns = list(PATTERNS)
    elif args.patterns == ["held_out"]:
        patterns = list(HELD_OUT_PATTERNS)
    else:
        patterns = args.patterns

    output_dir = args.output_dir or f"results/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_benchmark(args.agents, patterns, args.seeds, args.episodes_per_seed, output_dir)


if __name__ == "__main__":
    main()
