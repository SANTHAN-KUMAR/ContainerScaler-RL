"""
Additional Baselines — Comparison agents beyond Kubernetes HPA.

Three baselines that bracket performance from below and above:

  1. **FixedReplica** — Static allocation (no scaling at all).
     Tests whether *any* autoscaling is beneficial.

  2. **ReactiveThreshold** — Simple CPU threshold rules.
     Tests whether a trivial policy is "good enough".

  3. **OracleScaler** — Has perfect future knowledge of the workload.
     Provides an *upper bound* on performance — no learned agent can beat it.

All baselines expose ``act(state) → delta`` and ``reset()`` for compatibility
with the existing ``run_episode()`` harness.
"""

from __future__ import annotations

import math
from typing import Sequence

from src.safety.safety_filter import ClusterState


class FixedReplicaBaseline:
    """Always maintains a fixed number of replicas — zero scaling.

    This is the "do nothing" baseline.  If RL or HPA can't beat this,
    autoscaling provides no value.

    Parameters
    ----------
    fixed_replicas : int
        Number of replicas to maintain (default 5 = a reasonable middle ground).
    """

    def __init__(self, fixed_replicas: int = 5) -> None:
        self.fixed_replicas = fixed_replicas

    def act(self, state: ClusterState) -> int:
        """Return delta to reach the fixed replica count."""
        return self.fixed_replicas - state.replicas

    def reset(self) -> None:
        """No state to reset."""
        pass

    def __repr__(self) -> str:
        return f"FixedReplicaBaseline(replicas={self.fixed_replicas})"


class ReactiveThresholdBaseline:
    """Simple CPU-based threshold scaling — the simplest reactive policy.

    Rules:
      - CPU > high_threshold → scale up by ``scale_up_step``
      - CPU < low_threshold  → scale down by ``scale_down_step``
      - otherwise            → do nothing

    No stabilization window, no smoothing.  Tests whether a 3-line
    if/elif/else is "good enough" to render RL unnecessary.

    Parameters
    ----------
    high_threshold : float
        CPU utilization above which we scale up (default 0.80).
    low_threshold : float
        CPU utilization below which we scale down (default 0.30).
    scale_up_step : int
        Replicas to add per scale-up action (default 2).
    scale_down_step : int
        Replicas to remove per scale-down action (default 1).
    min_replicas : int
        Minimum replica count.
    max_replicas : int
        Maximum replica count.
    """

    def __init__(
        self,
        high_threshold: float = 0.80,
        low_threshold: float = 0.30,
        scale_up_step: int = 2,
        scale_down_step: int = 1,
        min_replicas: int = 2,
        max_replicas: int = 30,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.scale_up_step = scale_up_step
        self.scale_down_step = scale_down_step
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas

    def act(self, state: ClusterState) -> int:
        """Compute delta based on CPU thresholds."""
        if state.cpu_util > self.high_threshold:
            delta = self.scale_up_step
        elif state.cpu_util < self.low_threshold:
            delta = -self.scale_down_step
        else:
            delta = 0

        # Clamp to replica bounds
        new_replicas = state.replicas + delta
        new_replicas = max(self.min_replicas, min(self.max_replicas, new_replicas))
        return new_replicas - state.replicas

    def reset(self) -> None:
        """No state to reset."""
        pass

    def __repr__(self) -> str:
        return (
            f"ReactiveThresholdBaseline("
            f"high={self.high_threshold}, low={self.low_threshold})"
        )


class OracleScaler:
    """Perfect-future-knowledge scaler — the theoretical upper bound.

    Pre-computes the optimal replica count for each step by looking at
    the *entire* future workload.  No real agent can beat this.

    The oracle plans replicas as:
        ideal[t] = ceil(request_rate[t+lookahead] / per_pod_capacity)
    clamped to [min_replicas, max_replicas], with cold-start-aware
    pre-provisioning.

    Parameters
    ----------
    future_rates : sequence of float
        The full request rate trajectory for the episode (120 values).
    per_pod_capacity : float
        Requests per second each pod can handle.
    cold_start_steps : int
        How many steps ahead to pre-provision pods (accounts for cold start).
    min_replicas : int
        Minimum replica count.
    max_replicas : int
        Maximum replica count.
    """

    def __init__(
        self,
        future_rates: Sequence[float],
        per_pod_capacity: float = 20.0,
        cold_start_steps: int = 2,
        min_replicas: int = 2,
        max_replicas: int = 30,
    ) -> None:
        self.per_pod_capacity = per_pod_capacity
        self.cold_start_steps = cold_start_steps
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas

        # Pre-compute the optimal plan
        rates = list(future_rates)
        self._plan: list[int] = []
        for t in range(len(rates)):
            # Look ahead by cold_start_steps
            lookahead_t = min(t + cold_start_steps, len(rates) - 1)
            future_rate = max(rates[t], rates[lookahead_t])
            ideal = math.ceil(future_rate / per_pod_capacity)
            ideal = max(min_replicas, min(max_replicas, ideal))
            self._plan.append(ideal)

        self._step = 0

    def act(self, state: ClusterState) -> int:
        """Return the delta to reach the pre-computed optimal replica count."""
        if self._step < len(self._plan):
            target = self._plan[self._step]
            self._step += 1
        else:
            target = state.replicas

        delta = target - state.replicas
        # Clamp to ±3 (same action space as the RL agent)
        return max(-3, min(3, delta))

    def reset(self) -> None:
        """Reset step counter for a new episode."""
        self._step = 0

    def __repr__(self) -> str:
        return f"OracleScaler(plan_len={len(self._plan)}, lookahead={self.cold_start_steps})"
