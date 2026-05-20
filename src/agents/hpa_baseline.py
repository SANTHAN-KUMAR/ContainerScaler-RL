"""
RealisticHPA — Faithful replica of Kubernetes Horizontal Pod Autoscaler.

Implements the real HPA formula with a 3-minute scale-down stabilization
window (6 × 30s steps).  Serves as:
  1. Comparison baseline in experiments
  2. Fallback controller when the RL agent fails
"""

from __future__ import annotations

import math
from collections import deque

from src.safety.safety_filter import ClusterState


class RealisticHPA:
    """Kubernetes HPA with scale-down stabilization.

    Parameters
    ----------
    target_cpu : float
        Target CPU utilization (0–1).  Default 0.50 = 50%.
    min_replicas : int
        Minimum replica count.
    max_replicas : int
        Maximum replica count.
    stabilization_window : int
        Number of steps in the scale-down stabilization window.
        Default 6 (6 × 30s = 3 minutes, matching real K8s behavior).
    """

    def __init__(
        self,
        target_cpu: float = 0.50,
        min_replicas: int = 2,
        max_replicas: int = 30,
        stabilization_window: int = 6,
    ) -> None:
        self.target_cpu = target_cpu
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self._window_size = stabilization_window
        self.replica_history: deque[int] = deque(maxlen=stabilization_window)

    def act(self, state: ClusterState) -> int:
        """Compute the replica delta using the HPA formula.

        Parameters
        ----------
        state : ClusterState
            Current cluster state.

        Returns
        -------
        int
            Replica delta (desired - current).
        """
        # HPA formula: desired = ceil(current × cpu_util / target_cpu)
        cpu = max(state.cpu_util, 0.001)  # avoid zero
        desired = math.ceil(state.replicas * cpu / self.target_cpu)
        desired = max(self.min_replicas, min(self.max_replicas, desired))

        # Record in stabilization window
        self.replica_history.append(desired)

        # Scale-down uses max of recent window (real K8s behavior)
        if desired < state.replicas and self.replica_history:
            desired = max(self.replica_history)

        return int(desired - state.replicas)

    def reset(self) -> None:
        """Clear stabilization window for a new episode/run."""
        self.replica_history.clear()

    def __repr__(self) -> str:
        return (
            f"RealisticHPA(target_cpu={self.target_cpu}, "
            f"window={self._window_size})"
        )
