"""
MetricsLogger — Records data during live experiments.

Logs all state observations, agent proposals, and safety actions
to a CSV file for post-hoc analysis and plotting.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.safety.safety_filter import ClusterState

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Logs metrics during a live cluster run.

    Parameters
    ----------
    log_dir : str | Path
        Directory to write log files to.
    prefix : str
        Prefix for the log files.
    """

    def __init__(self, log_dir: str | Path = "./logs/live", prefix: str = "run") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        self.csv_path = self.log_dir / f"{prefix}_{timestamp}.csv"
        self.json_path = self.log_dir / f"{prefix}_{timestamp}_meta.json"

        self.records: list[dict[str, Any]] = []
        self.fieldnames = [
            "timestamp", "step", "p99_latency", "request_rate", 
            "replicas", "pending_pods", "cpu_util", "mem_util",
            "queue_depth", "proposed_delta", "safe_delta", 
            "cost_rate", "sla_breach", "source"
        ]

        # Initialize CSV
        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            
        logger.info(f"Logging metrics to {self.csv_path}")

    def log(
        self, 
        step: int, 
        state: ClusterState, 
        proposed_delta: int, 
        safe_delta: int, 
        sla_target: float = 200.0,
        source: str = "rl",
    ) -> None:
        """Record a single step's metrics.

        Parameters
        ----------
        step : int
            Current simulation/live step.
        state : ClusterState
            Parsed state from the observer.
        proposed_delta : int
            Delta proposed by the agent.
        safe_delta : int
            Delta applied after the safety filter.
        sla_target : float
            Latency target to determine SLA breach.
        source : str
            Source of the decision ("rl" or "hpa").
        """
        record = {
            "timestamp": time.time(),
            "step": step,
            "p99_latency": state.p99_latency,
            "request_rate": state.request_rate,
            "replicas": state.replicas,
            "pending_pods": state.pending_pods,
            "cpu_util": state.cpu_util,
            "mem_util": state.mem_util,
            "queue_depth": state.queue_depth,
            "proposed_delta": proposed_delta,
            "safe_delta": safe_delta,
            "cost_rate": state.cost_rate,
            "sla_breach": state.p99_latency > sla_target,
            "source": source,
        }
        self.records.append(record)

        # Append to CSV
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(record)

    def save_meta(self, metadata: dict[str, Any]) -> None:
        """Save run metadata to a JSON file."""
        with open(self.json_path, "w") as f:
            json.dump(metadata, f, indent=2)
