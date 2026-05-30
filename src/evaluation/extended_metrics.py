"""
Extended Metrics — Statistically rigorous evaluation metrics.

Supplements the base ``metrics.py`` with:
  - Effect sizes (Cohen's d)
  - Bootstrap confidence intervals
  - Bonferroni-corrected p-values
  - Domain-specific RL-autoscaler metrics (reaction time, settling time, etc.)

All functions are pure — no side effects, no global state.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


# ======================================================================
# Statistical tests & effect sizes
# ======================================================================

def cohens_d(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """Compute Cohen's d effect size between two independent samples.

    Interpretation (conventional thresholds):
      |d| < 0.2  → negligible
      0.2–0.5    → small
      0.5–0.8    → medium
      > 0.8      → large

    Returns 0.0 if either group has fewer than 2 observations.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return 0.0

    na, nb = len(a), len(b)
    # Pooled standard deviation
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    pooled_std = math.sqrt(max(pooled_var, 1e-12))
    return float((a.mean() - b.mean()) / pooled_std)


def effect_size_interpretation(d: float) -> str:
    """Return a human-readable interpretation of Cohen's d."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def bootstrap_ci(
    data: Sequence[float],
    statistic: str = "mean",
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : sequence of float
        Observed values.
    statistic : str
        ``"mean"`` or ``"median"``.
    confidence : float
        Confidence level (default 0.95 = 95% CI).
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    tuple[float, float, float]
        (point_estimate, ci_lower, ci_upper)
    """
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) < 2:
        val = float(arr[0]) if len(arr) == 1 else 0.0
        return val, val, val

    rng = np.random.default_rng(seed)
    stat_fn = np.mean if statistic == "mean" else np.median
    point = float(stat_fn(arr))

    # Bootstrap resamples
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats[i] = stat_fn(sample)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point, ci_lower, ci_upper


def bonferroni_correct(p_values: Sequence[float]) -> list[float]:
    """Apply Bonferroni correction to a list of p-values.

    Prevents false positives when running multiple statistical tests.
    """
    n = len(p_values)
    if n == 0:
        return []
    return [min(1.0, p * n) for p in p_values]


def mann_whitney_with_effect_size(
    group_a: Sequence[float],
    group_b: Sequence[float],
) -> dict[str, float | str]:
    """Mann-Whitney U test with Cohen's d effect size and interpretation.

    Returns
    -------
    dict
        Keys: ``u_stat``, ``p_value``, ``cohens_d``, ``effect_interpretation``.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)

    if len(a) < 2 or len(b) < 2:
        return {
            "u_stat": 0.0,
            "p_value": 1.0,
            "cohens_d": 0.0,
            "effect_interpretation": "insufficient data",
        }

    try:
        u_stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        u_stat, p_value = 0.0, 1.0

    d = cohens_d(a, b)
    return {
        "u_stat": float(u_stat),
        "p_value": float(p_value),
        "cohens_d": float(d),
        "effect_interpretation": effect_size_interpretation(d),
    }


# ======================================================================
# Domain-specific autoscaler metrics
# ======================================================================

def compute_reaction_time(
    request_rates: Sequence[float],
    replicas: Sequence[float | int],
    spike_threshold: float = 1.3,
) -> float:
    """Steps from traffic spike detection to first scale-up.

    A 'spike' is defined as request_rate[t] / request_rate[t-1] > spike_threshold.
    Reaction time is the number of steps between the spike and the first
    subsequent increase in replica count.

    Returns NaN if no spike is detected.
    """
    rates = np.asarray(request_rates, dtype=np.float64)
    reps = np.asarray(replicas, dtype=np.float64)

    reaction_times: list[int] = []

    for t in range(1, len(rates)):
        if rates[t - 1] > 0 and rates[t] / rates[t - 1] > spike_threshold:
            # Found a spike — find the next scale-up
            for dt in range(t, len(reps)):
                if reps[dt] > reps[t - 1]:
                    reaction_times.append(dt - t)
                    break

    return float(np.mean(reaction_times)) if reaction_times else float("nan")


def compute_over_provisioning_ratio(
    request_rates: Sequence[float],
    replicas: Sequence[float | int],
    per_pod_capacity: float = 20.0,
) -> float:
    """Average ratio of actual replicas to ideal replicas.

    ideal_replicas = ceil(request_rate / per_pod_capacity)
    ratio > 1.0 means over-provisioned (wasting money).
    ratio < 1.0 means under-provisioned (risking SLA).
    """
    rates = np.asarray(request_rates, dtype=np.float64)
    reps = np.asarray(replicas, dtype=np.float64)

    ratios: list[float] = []
    for r, rep in zip(rates, reps):
        ideal = max(1.0, math.ceil(r / per_pod_capacity))
        ratios.append(rep / ideal)

    return float(np.mean(ratios)) if ratios else 1.0


def compute_under_provisioning_ratio(
    cpu_utils: Sequence[float],
    threshold: float = 0.9,
) -> float:
    """Fraction of steps where CPU utilization exceeds threshold.

    High values indicate the agent isn't scaling up fast enough.
    """
    if not cpu_utils:
        return 0.0
    over = sum(1 for u in cpu_utils if u > threshold)
    return over / len(cpu_utils)


def compute_settling_time(
    latencies: Sequence[float],
    sla_target: float = 200.0,
    tolerance: float = 0.05,
) -> float:
    """Steps from first SLA violation to sustained compliance.

    Sustained = latency stays within ``tolerance × sla_target`` of the target
    for ≥ 5 consecutive steps.

    Returns NaN if no violation occurs.
    """
    lats = np.asarray(latencies, dtype=np.float64)
    threshold = sla_target * (1.0 + tolerance)

    # Find first violation
    first_violation = -1
    for i, lat in enumerate(lats):
        if lat > sla_target:
            first_violation = i
            break

    if first_violation < 0:
        return float("nan")

    # Find sustained compliance after violation
    window = 5
    for i in range(first_violation, len(lats) - window + 1):
        if all(lats[j] <= threshold for j in range(i, i + window)):
            return float(i - first_violation)

    return float(len(lats) - first_violation)  # never settled


def compute_cost_efficiency_pct(
    rl_costs: Sequence[float],
    hpa_costs: Sequence[float],
) -> float:
    """Percentage cost savings of RL over HPA.

    Positive = RL is cheaper. Negative = RL is more expensive.
    """
    rl_avg = float(np.mean(rl_costs)) if rl_costs else 0.0
    hpa_avg = float(np.mean(hpa_costs)) if hpa_costs else 0.0
    if hpa_avg == 0:
        return 0.0
    return (hpa_avg - rl_avg) / hpa_avg * 100.0


def compute_action_entropy(actions: Sequence[int], n_actions: int = 7) -> float:
    """Shannon entropy of the agent's action distribution.

    Higher entropy = more diverse behavior (explores the action space).
    Maximum entropy = log2(n_actions) ≈ 2.81 for 7 actions.
    """
    if not actions:
        return 0.0
    counts = np.zeros(n_actions)
    for a in actions:
        if 0 <= a < n_actions:
            counts[a] += 1
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    # Filter out zero probabilities to avoid log(0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_jain_fairness(values: Sequence[float]) -> float:
    """Jain's fairness index — measures consistency across steps.

    Returns a value in (0, 1]. 1.0 = perfectly consistent.
    Useful for measuring whether resource allocation is stable or erratic.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0 or arr.sum() == 0:
        return 1.0
    n = len(arr)
    return float(arr.sum() ** 2 / (n * (arr ** 2).sum()))


def compute_coefficient_of_variation(values: Sequence[float]) -> float:
    """Coefficient of variation (CV) — relative variability.

    CV > 0.15 (15%) on a key metric flags instability.
    Returns 0.0 if mean is zero.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    mean = arr.mean()
    if abs(mean) < 1e-12:
        return 0.0
    return float(arr.std(ddof=1) / abs(mean))
