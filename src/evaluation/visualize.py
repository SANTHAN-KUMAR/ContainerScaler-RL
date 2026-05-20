"""
Visualization — Plotting utilities for experiments.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_live_run(csv_path: str | Path, save_dir: str | Path = "./plots") -> None:
    """Plot metrics from a live run CSV log."""
    csv_path = Path(csv_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"Cannot plot: {csv_path} not found.")
        return
        
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # 1. Traffic
    axs[0].plot(df["step"], df["request_rate"], color="blue", label="Request Rate (rps)")
    axs[0].set_ylabel("Requests/sec")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title(f"Live Run: {csv_path.stem}")
    
    # 2. Replicas
    axs[1].plot(df["step"], df["replicas"], color="green", label="Ready Replicas")
    axs[1].plot(df["step"], df["replicas"] + df["pending_pods"], color="lightgreen", linestyle="--", label="Total (inc. pending)")
    axs[1].set_ylabel("Pods")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # 3. Latency
    axs[2].plot(df["step"], df["p99_latency"], color="red", label="P99 Latency (ms)")
    axs[2].axhline(y=200, color="darkred", linestyle="--", label="SLA Target (200ms)")
    axs[2].set_ylabel("Latency (ms)")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    # 4. Deltas & Safety
    axs[3].plot(df["step"], df["proposed_delta"], color="orange", label="Proposed Delta", alpha=0.5)
    axs[3].plot(df["step"], df["safe_delta"], color="black", label="Applied Delta (Safe)")
    axs[3].set_ylabel("Replica Delta")
    axs[3].set_xlabel("Step (30s intervals)")
    axs[3].legend()
    axs[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = save_dir / f"{csv_path.stem}_plot.png"
    plt.savefig(out_path)
    logger.info(f"Plot saved to {out_path}")
    plt.close()
