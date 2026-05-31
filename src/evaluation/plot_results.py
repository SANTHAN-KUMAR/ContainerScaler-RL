"""
Plot Results — Publication-quality visualization from benchmark data.

Generates clean, interpretable charts from benchmark_runner output:
  1. Metric comparison heatmap
  2. Per-pattern radar charts
  3. Cost vs SLA Pareto front
  4. Trajectory overlays (per-pattern)
  5. Seed stability violin plots
  6. Overall ranking bar chart
  7. Latency distribution comparison

All plots use a consistent dark professional theme with high DPI.

Usage:
    python -m src.evaluation.plot_results --results-dir results/benchmark_20260529
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ======================================================================
# Theme configuration
# ======================================================================

# Professional color palette
COLORS = {
    "RL-PPO": "#4FC3F7",
    "RL-DQN": "#81C784",
    "RL-QRDQN": "#FFB74D",
    "HPA": "#E57373",
    "Fixed-5": "#CE93D8",
    "Fixed-10": "#A1887F",
    "Reactive": "#90A4AE",
    "Oracle": "#FFD54F",
}

def _get_color(agent_name: str) -> str:
    """Get color for an agent, with fallback."""
    for key, color in COLORS.items():
        if key in agent_name:
            return color
    # Fallback palette
    fallback = ["#4DD0E1", "#AED581", "#FF8A65", "#BA68C8", "#4DB6AC"]
    return fallback[hash(agent_name) % len(fallback)]


def _setup_style():
    """Configure matplotlib for clean, professional plots."""
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


# ======================================================================
# Plot generators
# ======================================================================

def plot_overall_ranking(df: pd.DataFrame, save_path: Path) -> None:
    """Horizontal bar chart ranking agents by composite score."""
    _setup_style()
    ranking = (df.groupby("agent")["composite"]
               .agg(["mean", "std"])
               .sort_values("mean", ascending=True))

    fig, ax = plt.subplots(figsize=(10, max(4, len(ranking) * 0.8)))
    colors = [_get_color(a) for a in ranking.index]
    bars = ax.barh(ranking.index, ranking["mean"], xerr=ranking["std"],
                   color=colors, edgecolor="white", linewidth=0.5,
                   capsize=4, error_kw={"elinewidth": 1.5, "capthick": 1.5})

    # Add value labels
    for bar, (_, row) in zip(bars, ranking.iterrows()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{row['mean']:.3f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Composite Score (higher = better)", fontsize=12)
    ax.set_title("Overall Agent Ranking", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, min(1.0, ranking["mean"].max() * 1.3))
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_metric_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    """Heatmap of agents × key metrics, normalized for comparison."""
    _setup_style()
    metrics = ["sla_compliance", "avg_cost", "avg_latency", "churn",
               "cpu_util", "reaction_time", "composite"]
    metric_labels = ["SLA %", "Avg Cost", "Avg Latency", "Churn",
                     "CPU Util", "Reaction Time", "Composite"]
    # Higher-is-better flags (for coloring)
    higher_better = [True, False, False, False, False, False, True]

    agents = sorted(df["agent"].unique())
    data = np.zeros((len(agents), len(metrics)))

    for i, agent in enumerate(agents):
        for j, m in enumerate(metrics):
            vals = df[df["agent"] == agent][m].dropna()
            data[i, j] = vals.mean() if len(vals) > 0 else 0

    # Normalize each column to [0, 1]
    norm_data = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 1e-9:
            norm_data[:, j] = (col - col_min) / (col_max - col_min)
            if not higher_better[j]:
                norm_data[:, j] = 1.0 - norm_data[:, j]
        else:
            norm_data[:, j] = 0.5

    fig, ax = plt.subplots(figsize=(12, max(4, len(agents) * 0.7)))
    im = ax.imshow(norm_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents)

    # Annotate cells with raw values
    for i in range(len(agents)):
        for j in range(len(metrics)):
            val = data[i, j]
            fmt = f"{val:.1f}" if val > 10 else f"{val:.3f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=9,
                    color="black" if norm_data[i, j] > 0.5 else "white")

    ax.set_title("Agent Performance Heatmap (green = better)",
                 fontsize=14, fontweight="bold", pad=15)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Normalized Score")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_cost_vs_sla_pareto(df: pd.DataFrame, save_path: Path) -> None:
    """Scatter plot of cost vs SLA compliance with Pareto front."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    agents = sorted(df["agent"].unique())
    for agent in agents:
        subset = df[df["agent"] == agent]
        color = _get_color(agent)
        ax.scatter(subset["avg_cost"], subset["sla_compliance"],
                   c=color, label=agent, alpha=0.4, s=30, edgecolors="white", linewidth=0.3)
        # Mean marker
        ax.scatter(subset["avg_cost"].mean(), subset["sla_compliance"].mean(),
                   c=color, s=200, marker="*", edgecolors="white", linewidth=1.5, zorder=5)

    ax.set_xlabel("Average Cost ($/hr)", fontsize=12)
    ax.set_ylabel("SLA Compliance (%)", fontsize=12)
    ax.set_title("Cost vs SLA Compliance (★ = mean per agent)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower left", framealpha=0.8, fontsize=9)
    ax.axhline(y=95, color="#E57373", linestyle="--", alpha=0.5, label="95% SLA target")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_per_pattern_comparison(df: pd.DataFrame, save_dir: Path) -> None:
    """Bar charts comparing agents on each workload pattern."""
    _setup_style()
    patterns = sorted(df["pattern"].unique())
    agents = sorted(df["agent"].unique())
    metrics = [("sla_compliance", "SLA Compliance (%)", True),
               ("avg_cost", "Avg Cost ($/hr)", False),
               ("composite", "Composite Score", True)]

    for metric, label, higher_better in metrics:
        fig, axes = plt.subplots(1, len(patterns), figsize=(4 * len(patterns), 5),
                                 sharey=True)
        if len(patterns) == 1:
            axes = [axes]

        for ax, pattern in zip(axes, patterns):
            means, stds, colors = [], [], []
            agent_labels = []
            for agent in agents:
                subset = df[(df["agent"] == agent) & (df["pattern"] == pattern)]
                if len(subset) == 0:
                    continue
                means.append(subset[metric].mean())
                stds.append(subset[metric].std())
                colors.append(_get_color(agent))
                agent_labels.append(agent)

            x = np.arange(len(agent_labels))
            ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
                   linewidth=0.5, capsize=3, width=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(agent_labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(pattern, fontsize=11, fontweight="bold")
            if ax == axes[0]:
                ax.set_ylabel(label, fontsize=10)

        fig.suptitle(f"{label} by Pattern", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(save_dir / f"pattern_{metric}.png")
        plt.close(fig)
        logger.info("Saved: pattern_%s.png", metric)


def plot_trajectory_overlay(trajectories: dict, save_dir: Path) -> None:
    """Overlay trajectory plots for all agents on each pattern."""
    _setup_style()
    patterns = set()
    for key, traj in trajectories.items():
        patterns.add(traj["pattern"])

    for pattern in sorted(patterns):
        # Get all agents for this pattern
        pat_trajs = {k: v for k, v in trajectories.items() if v["pattern"] == pattern}
        if not pat_trajs:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        for key, traj in pat_trajs.items():
            agent = traj["agent"]
            color = _get_color(agent)
            steps = range(len(traj["replicas"]))

            # Replicas
            axes[0].plot(steps, traj["replicas"], color=color, label=agent,
                         alpha=0.8, linewidth=1.5)
            # Latency
            axes[1].plot(steps, traj["latencies"], color=color, label=agent,
                         alpha=0.8, linewidth=1.5)
            # Cost
            axes[2].plot(steps, traj["costs"], color=color, label=agent,
                         alpha=0.8, linewidth=1.5)

        # Request rate on secondary axis (from first trajectory)
        first_traj = next(iter(pat_trajs.values()))
        ax_rps = axes[0].twinx()
        ax_rps.fill_between(range(len(first_traj["request_rates"])),
                            first_traj["request_rates"],
                            alpha=0.15, color="white", label="RPS")
        ax_rps.set_ylabel("Request Rate (rps)", fontsize=10, color="#888")
        ax_rps.tick_params(axis="y", colors="#888")

        axes[0].set_ylabel("Replicas", fontsize=11)
        axes[0].legend(loc="upper left", fontsize=8, framealpha=0.8)
        axes[1].set_ylabel("P99 Latency (ms)", fontsize=11)
        axes[1].axhline(y=200, color="#E57373", linestyle="--", alpha=0.7, linewidth=1)
        axes[1].legend(loc="upper left", fontsize=8, framealpha=0.8)
        axes[2].set_ylabel("Cost ($/hr)", fontsize=11)
        axes[2].set_xlabel("Step (30s intervals)", fontsize=11)
        axes[2].legend(loc="upper left", fontsize=8, framealpha=0.8)

        fig.suptitle(f"Trajectory: {pattern}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(save_dir / f"{pattern}_trajectory.png")
        plt.close(fig)
        logger.info("Saved: %s_trajectory.png", pattern)


def plot_stability_violin(df: pd.DataFrame, save_path: Path) -> None:
    """Violin plot of composite score distribution across seeds."""
    _setup_style()
    agents = sorted(df["agent"].unique())
    fig, ax = plt.subplots(figsize=(max(8, len(agents) * 1.5), 6))

    data = [df[df["agent"] == a]["composite"].dropna().values for a in agents]
    colors = [_get_color(a) for a in agents]

    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    parts["cmeans"].set_color("white")
    parts["cmedians"].set_color("#FFD54F")
    parts["cmins"].set_color("#e0e0e0")
    parts["cmaxes"].set_color("#e0e0e0")
    parts["cbars"].set_color("#e0e0e0")

    ax.set_xticks(range(1, len(agents) + 1))
    ax.set_xticklabels(agents, rotation=30, ha="right")
    ax.set_ylabel("Composite Score", fontsize=12)
    ax.set_title("Score Distribution Across Seeds (stability check)",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_latency_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """Box plot comparing latency distributions across agents."""
    _setup_style()
    agents = sorted(df["agent"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    lat_metrics = [("avg_latency", "Average Latency (ms)"),
                   ("p99_latency", "P99 Latency (ms)"),
                   ("max_latency", "Max Latency (ms)")]

    for ax, (metric, label) in zip(axes, lat_metrics):
        data = [df[df["agent"] == a][metric].dropna().values for a in agents]
        bp = ax.boxplot(data, patch_artist=True, labels=agents)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(_get_color(agents[i]))
            patch.set_alpha(0.7)
        ax.set_ylabel(label, fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        if metric in ("p99_latency", "max_latency"):
            ax.axhline(y=200, color="#E57373", linestyle="--", alpha=0.5)

    fig.suptitle("Latency Distribution Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_reaction_and_settling(df: pd.DataFrame, save_path: Path) -> None:
    """Box plots of reaction time and settling time."""
    _setup_style()
    agents = sorted(df["agent"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (metric, label) in zip(axes, [
        ("reaction_time", "Reaction Time (steps)"),
        ("settling_time", "Settling Time (steps)"),
    ]):
        data = [df[df["agent"] == a][metric].dropna().values for a in agents]
        # Filter out empty arrays
        valid_agents = [a for a, d in zip(agents, data) if len(d) > 0]
        valid_data = [d for d in data if len(d) > 0]
        if not valid_data:
            continue
        bp = ax.boxplot(valid_data, patch_artist=True, labels=valid_agents)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(_get_color(valid_agents[i]))
            patch.set_alpha(0.7)
        ax.set_ylabel(label, fontsize=10)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Responsiveness Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)

def plot_latency_cdf(trajectories: dict, save_path: Path, sla_target: float = 200.0) -> None:
    """Cumulative Distribution Function of per-step latencies.

    Shows the full distribution of individual step latencies, not episode
    averages. This exposes tail behavior that box plots and means hide.
    The vertical SLA line makes it clear what fraction of requests breach.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    agent_latencies: dict[str, list[float]] = {}
    for key, traj in trajectories.items():
        agent = traj["agent"]
        lats = traj.get("latencies", [])
        if agent not in agent_latencies:
            agent_latencies[agent] = []
        agent_latencies[agent].extend(lats)

    for agent in sorted(agent_latencies.keys()):
        lats = sorted(agent_latencies[agent])
        n = len(lats)
        if n == 0:
            continue
        cdf = np.arange(1, n + 1) / n * 100
        color = _get_color(agent)
        ax.plot(lats, cdf, color=color, label=agent, linewidth=2, alpha=0.85)

    # SLA target line
    ax.axvline(x=sla_target, color="#E57373", linestyle="--", linewidth=2,
               alpha=0.8, label=f"SLA Target ({sla_target}ms)")

    ax.set_xlabel("P99 Latency (ms)", fontsize=12)
    ax.set_ylabel("Cumulative % of Steps", fontsize=12)
    ax.set_title("Latency CDF — Full Distribution of Per-Step Latencies",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", framealpha=0.8, fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


# ======================================================================
# Main entry point
# ======================================================================

def generate_all_plots(results_dir: Path | str) -> None:
    """Generate all plots from benchmark results directory."""
    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = plots_dir / "trajectory_overlays"
    traj_dir.mkdir(exist_ok=True)

    # Load data
    csv_path = results_dir / "raw_results.csv"
    if not csv_path.exists():
        logger.error("No raw_results.csv found in %s", results_dir)
        return

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d episodes from %s", len(df), csv_path)

    # Generate plots
    plot_overall_ranking(df, plots_dir / "overall_ranking.png")
    plot_metric_heatmap(df, plots_dir / "metric_heatmap.png")
    plot_cost_vs_sla_pareto(df, plots_dir / "cost_vs_sla_pareto.png")
    plot_per_pattern_comparison(df, plots_dir)
    plot_stability_violin(df, plots_dir / "stability_violin.png")
    plot_latency_distribution(df, plots_dir / "latency_distribution.png")
    plot_reaction_and_settling(df, plots_dir / "responsiveness.png")

    # Trajectory overlays (if trajectories.json exists)
    traj_path = results_dir / "trajectories.json"
    if traj_path.exists():
        with open(traj_path) as f:
            trajectories = json.load(f)
        plot_trajectory_overlay(trajectories, traj_dir)
        plot_latency_cdf(trajectories, plots_dir / "latency_cdf.png")
    else:
        logger.warning("No trajectories.json found, skipping trajectory plots")

    logger.info("All plots saved to %s", plots_dir)


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate plots from benchmark results.")
    parser.add_argument("--results-dir", required=True, help="Path to benchmark results directory")
    args = parser.parse_args()
    generate_all_plots(args.results_dir)


if __name__ == "__main__":
    main()
