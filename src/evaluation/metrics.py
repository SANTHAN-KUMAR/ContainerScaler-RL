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
