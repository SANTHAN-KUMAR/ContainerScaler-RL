"""
Metrics — Aggregation functions for experiment results.
"""

from __future__ import annotations



def compute_sla_compliance(latencies: list[float], sla_target: float = 200.0) -> float:
    """Percentage of steps where latency is <= SLA target."""
    if not latencies:
        return 100.0
    compliant = sum(1 for lat in latencies if lat <= sla_target)
    return (compliant / len(latencies)) * 100.0


def compute_average_cost(costs: list[float]) -> float:
    """Average cost per hour."""
    if not costs:
        return 0.0
    return sum(costs) / len(costs)


def compute_churn(replicas: list[int]) -> int:
    """Total absolute replica changes over the run."""
    if len(replicas) < 2:
        return 0
    churn = 0
    for i in range(1, len(replicas)):
        churn += abs(replicas[i] - replicas[i - 1])
    return churn


def compute_max_latency(latencies: list[float]) -> float:
    """Maximum latency encountered."""
    if not latencies:
        return 0.0
    return max(latencies)


def compute_average_latency(latencies: list[float]) -> float:
    """Average latency encountered."""
    if not latencies:
        return 0.0
    return sum(latencies) / len(latencies)


def compute_percentile_latency(latencies: list[float], percentile: float = 95.0) -> float:
    """Nth percentile of latency."""
    if not latencies:
        return 0.0
    latencies_sorted = sorted(latencies)
    idx = int((percentile / 100.0) * len(latencies_sorted))
    idx = min(idx, len(latencies_sorted) - 1)
    return latencies_sorted[idx]


def compute_violation_severity(latencies: list[float], sla_target: float = 200.0) -> float:
    """Average magnitude of latency over the SLA target."""
    violations = [lat - sla_target for lat in latencies if lat > sla_target]
    if not violations:
        return 0.0
    return sum(violations) / len(violations)


def compute_mean_utilization(utils: list[float]) -> float:
    """Average resource utilization."""
    if not utils:
        return 0.0
    return sum(utils) / len(utils)


def compute_total_reward(rewards: list[float]) -> float:
    """Cumulative reward over the episode."""
    if not rewards:
        return 0.0
    return sum(rewards)
