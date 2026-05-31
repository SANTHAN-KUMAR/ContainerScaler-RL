"""
Feature Saliency Explainer — Perturbation-based black-box audit.

Answers: "What features does the RL agent actually look at?"

Method: For each observation dimension, perturb it by ±10% while holding
everything else constant, and measure how much the agent's output changes.
Features with high sensitivity = the agent relies on them heavily.

This is agent-agnostic — works on PPO, DQN, QRDQN, Ensemble, and baselines.

If an agent is exploiting a simulator quirk (e.g. relying on a feature
that wouldn't exist in production), this will expose it.

Usage:
    python -m src.evaluation.explain_agent --model ppo_autoscaler
    python -m src.evaluation.explain_agent --model ensemble
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.agents.agent import ContainerScaleAgent
from src.env.k8s_sim import K8sSimEnv
from src.env.workload import ALL_PATTERNS

logger = logging.getLogger(__name__)


# ======================================================================
# Observation Feature Labels
# ======================================================================

# Maps observation index → human-readable name and logical group
FEATURE_MAP: list[tuple[str, str]] = [
    ("cpu_util", "Resource"),                     # [0]
    ("mem_util", "Resource"),                     # [1]
    ("replicas_ratio", "Replica State"),          # [2]
    ("pending_pods", "Replica State"),            # [3]
    ("request_rate", "Traffic"),                  # [4]
    ("traffic_velocity", "Traffic"),              # [5]
    ("p99_latency", "Latency"),                   # [6]
    ("per_pod_capacity", "Resource"),             # [7]
    ("queue_depth", "Latency"),                   # [8]
    ("node0_cpu_avail", "Infrastructure"),        # [9]
    ("node0_mem_avail", "Infrastructure"),        # [10]
    ("node0_pod_count", "Infrastructure"),        # [11]
    ("node1_cpu_avail", "Infrastructure"),        # [12]
    ("node1_mem_avail", "Infrastructure"),        # [13]
    ("node1_pod_count", "Infrastructure"),        # [14]
    ("node2_cpu_avail", "Infrastructure"),        # [15]
    ("node2_mem_avail", "Infrastructure"),        # [16]
    ("node2_pod_count", "Infrastructure"),        # [17]
    ("time_sin", "Temporal"),                     # [18]
    ("time_cos", "Temporal"),                     # [19]
    ("cost_rate", "Cost"),                        # [20]
    ("prev_cpu_util", "Resource"),                # [21]
    ("traffic_accel", "Traffic"),                 # [22]
]

# Group colors for the bar chart
GROUP_COLORS = {
    "Traffic": "#4FC3F7",
    "Latency": "#E57373",
    "Resource": "#81C784",
    "Replica State": "#FFB74D",
    "Infrastructure": "#90A4AE",
    "Temporal": "#CE93D8",
    "Cost": "#FFD54F",
}


def _load_agent(model_path: str) -> Any:
    """Load an agent. Supports 'ensemble' and standard RL models."""
    if model_path == "ensemble":
        from src.agents.ensemble_agent import EnsembleMetaAgent
        return EnsembleMetaAgent()
    return ContainerScaleAgent(model_path=model_path)


def compute_feature_saliency(
    agent: Any,
    n_episodes: int = 10,
    perturbation: float = 0.10,
    seed_base: int = 3000,
) -> dict[str, Any]:
    """Compute perturbation-based feature importance.

    For each of the 23 observation features:
    1. Run a step and record the agent's action on the original observation.
    2. Perturb that single feature by +perturbation and -perturbation.
    3. Record how much the action changes.
    4. Average across all steps and episodes.

    Parameters
    ----------
    agent : Any
        Must have a ``decide(obs, step) -> int`` method.
    n_episodes : int
        Episodes to sample from.
    perturbation : float
        Perturbation magnitude (default 0.10 = ±10%).
    seed_base : int
        Starting seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``feature_importance`` (list of 23 floats),
        ``feature_names``, ``feature_groups``, ``n_samples``.
    """
    n_features = 23
    importance = np.zeros(n_features, dtype=np.float64)
    n_samples = 0

    for ep in range(n_episodes):
        pattern = ALL_PATTERNS[ep % len(ALL_PATTERNS)]
        env = K8sSimEnv(workload_pattern=pattern)
        obs, info = env.reset(seed=seed_base + ep)
        if hasattr(agent, "reset"):
            agent.reset()

        terminated = truncated = False
        step = 0

        while not (terminated or truncated):
            # Get the baseline action
            baseline_action = agent.decide(obs, step)

            # Perturb each feature independently
            for f_idx in range(n_features):
                orig_val = obs[f_idx]

                # Perturb up
                obs_up = obs.copy()
                delta = max(abs(orig_val) * perturbation, 0.01)
                obs_up[f_idx] = orig_val + delta
                action_up = agent.decide(obs_up, step)

                # Perturb down
                obs_down = obs.copy()
                obs_down[f_idx] = orig_val - delta
                action_down = agent.decide(obs_down, step)

                # Sensitivity = average absolute action change
                importance[f_idx] += (
                    abs(action_up - baseline_action) +
                    abs(action_down - baseline_action)
                ) / 2.0

            n_samples += 1

            # Step the environment forward (use original action)
            obs, reward, terminated, truncated, info = env.step(baseline_action + 3)
            step += 1

    # Normalize to [0, 1] range
    if n_samples > 0:
        importance /= n_samples
    max_imp = importance.max()
    if max_imp > 0:
        importance /= max_imp

    return {
        "feature_importance": importance.tolist(),
        "feature_names": [f[0] for f in FEATURE_MAP],
        "feature_groups": [f[1] for f in FEATURE_MAP],
        "n_samples": n_samples,
        "perturbation": perturbation,
    }


def compute_group_importance(saliency: dict[str, Any]) -> dict[str, float]:
    """Aggregate per-feature importance into logical groups."""
    groups: dict[str, list[float]] = {}
    for imp, (name, group) in zip(saliency["feature_importance"], FEATURE_MAP):
        if group not in groups:
            groups[group] = []
        groups[group].append(imp)

    return {g: float(np.mean(vals)) for g, vals in sorted(
        groups.items(), key=lambda x: np.mean(x[1]), reverse=True
    )}


def plot_feature_saliency(
    saliency: dict[str, Any], save_path: Path, model_name: str = "",
) -> None:
    """Generate a horizontal bar chart of feature importance."""
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "axes.grid": True,
        "grid.color": "#2a2a4a",
        "grid.alpha": 0.5,
        "text.color": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "font.family": "sans-serif",
        "font.size": 11,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#1a1a2e",
    })

    names = saliency["feature_names"]
    importances = saliency["feature_importance"]
    groups = saliency["feature_groups"]

    # Sort by importance
    order = np.argsort(importances)
    sorted_names = [names[i] for i in order]
    sorted_imp = [importances[i] for i in order]
    sorted_colors = [GROUP_COLORS.get(groups[i], "#888") for i in order]

    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(range(len(sorted_names)), sorted_imp, color=sorted_colors,
                   edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel("Relative Importance (0 = ignored, 1 = most relied upon)", fontsize=11)
    title = f"Feature Saliency Analysis"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Legend for groups
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.8)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_group_importance(group_imp: dict[str, float], save_path: Path, model_name: str = "") -> None:
    """Pie chart of importance by feature group."""
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "text.color": "#e0e0e0",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#1a1a2e",
    })

    groups = list(group_imp.keys())
    values = list(group_imp.values())
    colors = [GROUP_COLORS.get(g, "#888") for g in groups]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        values, labels=groups, colors=colors, autopct="%1.1f%%",
        startangle=140, textprops={"fontsize": 11},
    )
    for t in autotexts:
        t.set_color("black")
        t.set_fontweight("bold")

    title = "Feature Group Importance"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def explain(
    model_path: str,
    output_dir: str = "results/explainability",
    n_episodes: int = 10,
) -> dict[str, Any]:
    """Full explainability pipeline: compute saliency, plot, save."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_name = model_path.replace("_autoscaler", "").replace("_", " ").upper()
    if model_path == "ensemble":
        model_name = "Ensemble"

    print(f"\n{'='*60}")
    print(f"  Feature Saliency Analysis: {model_name}")
    print(f"{'='*60}")

    agent = _load_agent(model_path)
    print(f"  Running perturbation analysis ({n_episodes} episodes)...")

    saliency = compute_feature_saliency(agent, n_episodes=n_episodes)
    group_imp = compute_group_importance(saliency)

    # Print results
    print(f"\n  {'Feature':<20} | {'Group':<15} | {'Importance':>10}")
    print("  " + "-" * 52)
    order = np.argsort(saliency["feature_importance"])[::-1]
    for i in order:
        name = saliency["feature_names"][i]
        group = saliency["feature_groups"][i]
        imp = saliency["feature_importance"][i]
        bar = "█" * int(imp * 20)
        print(f"  {name:<20} | {group:<15} | {imp:>8.3f}  {bar}")

    print(f"\n  Group Summary:")
    for g, v in group_imp.items():
        print(f"    {g:<20}: {v:.3f}")

    # Save plots
    plot_feature_saliency(saliency, out / "feature_saliency.png", model_name)
    plot_group_importance(group_imp, out / "group_importance.png", model_name)

    # Save raw data
    (out / "saliency_data.json").write_text(
        json.dumps({"saliency": saliency, "group_importance": group_imp},
                   indent=2, default=str))

    print(f"\n  Results saved to {out}/")
    return saliency


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Feature Saliency Explainer")
    parser.add_argument("--model", default="ppo_autoscaler", help="Model path or 'ensemble'")
    parser.add_argument("--output-dir", default="results/explainability")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    explain(args.model, args.output_dir, args.episodes)


if __name__ == "__main__":
    main()
