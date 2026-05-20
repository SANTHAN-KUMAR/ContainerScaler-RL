"""
SafetyFilter — Hard-coded invariant rules on every scaling action.

Never trained. Never overridden by the agent. Pure Python, zero dependencies.

Rules (applied in order):
  1. Absolute replica bounds [2, 30]
  2. No scale-down during high latency (> 0.8 × SLA target)
  3. Max delta per step ±3
  4. Rate limiting — minimum 1 step (30s) between actions
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClusterState:
    """Minimal view of cluster state needed by the safety filter.

    This can be constructed from the raw 22-dim observation vector
    or from a live PrometheusObserver snapshot.
    """
    replicas: int
    p99_latency: float
    cpu_util: float = 0.0
    request_rate: float = 0.0
    pending_pods: int = 0
    queue_depth: float = 0.0
    mem_util: float = 0.0
    cost_rate: float = 0.0

    @classmethod
    def from_obs(cls, obs, *, sla_target: float = 200.0, max_replicas: int = 30) -> "ClusterState":
        """Construct from the 22-dim observation vector (un-normalize)."""
        return cls(
            replicas=int(round(obs[2] * max_replicas)),
            p99_latency=float(obs[6] * 1000.0),
            cpu_util=float(obs[0]),
            request_rate=float(obs[4] * 500.0),
            pending_pods=int(round(obs[3] * 10.0)),
            queue_depth=float(obs[8] * 10000.0),
            mem_util=float(obs[1]),
            cost_rate=float(obs[20] * 2.0),
        )


class SafetyFilter:
    """Enforces hard safety invariants on every scaling action.

    Parameters
    ----------
    min_replicas : int
        Absolute minimum replica count.
    max_replicas : int
        Absolute maximum replica count.
    max_delta : int
        Maximum scaling step per action.
    min_interval_steps : int
        Minimum steps between non-zero actions (cooldown).
    sla_target : float
        P99 latency SLA target in ms.
    """

    def __init__(
        self,
        min_replicas: int = 2,
        max_replicas: int = 30,
        max_delta: int = 3,
        min_interval_steps: int = 1,
        sla_target: float = 200.0,
    ) -> None:
        self.MIN_REPLICAS = min_replicas
        self.MAX_REPLICAS = max_replicas
        self.MAX_DELTA = max_delta
        self.MIN_INTERVAL_STEPS = min_interval_steps
        self.sla_target = sla_target
        self.last_action_step: int = -min_interval_steps  # allow first action

    def check(self, state: ClusterState, proposed_delta: int, step: int) -> int:
        """Apply all 4 safety rules and return the (possibly modified) delta.

        Parameters
        ----------
        state : ClusterState
            Current cluster state snapshot.
        proposed_delta : int
            RL agent's proposed replica change (-3 to +3).
        step : int
            Current step number (for rate limiting).

        Returns
        -------
        int
            Safe delta — may be clipped, zeroed, or unchanged.
        """
        delta = int(proposed_delta)

        # Rule 1: Absolute replica bounds
        new_replicas = state.replicas + delta
        if new_replicas < self.MIN_REPLICAS:
            delta = self.MIN_REPLICAS - state.replicas
        elif new_replicas > self.MAX_REPLICAS:
            delta = self.MAX_REPLICAS - state.replicas

        # Rule 2: No scale-down during high latency
        if delta < 0 and state.p99_latency > 0.8 * self.sla_target:
            delta = 0

        # Rule 3: Max delta per step
        delta = max(-self.MAX_DELTA, min(self.MAX_DELTA, delta))

        # Rule 4: Rate limiting (cooldown)
        if delta != 0 and (step - self.last_action_step) < self.MIN_INTERVAL_STEPS:
            delta = 0

        # Track last action
        if delta != 0:
            self.last_action_step = step

        return delta

    def reset(self) -> None:
        """Reset cooldown timer for a new episode/run."""
        self.last_action_step = -self.MIN_INTERVAL_STEPS

    def __repr__(self) -> str:
        return (
            f"SafetyFilter(replicas=[{self.MIN_REPLICAS},{self.MAX_REPLICAS}], "
            f"max_delta={self.MAX_DELTA}, cooldown={self.MIN_INTERVAL_STEPS})"
        )
