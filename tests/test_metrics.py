"""Tests for metrics.py and extended_metrics.py."""

from __future__ import annotations

import math
import pytest
import numpy as np

from src.evaluation.metrics import (
    compute_average_cost,
    compute_average_latency,
    compute_churn,
    compute_max_latency,
    compute_mean_utilization,
    compute_percentile_latency,
    compute_sla_compliance,
    compute_total_reward,
    compute_violation_severity,
)
from src.evaluation.extended_metrics import (
    bootstrap_ci,
    bonferroni_correct,
    cohens_d,
    compute_action_entropy,
    compute_coefficient_of_variation,
    compute_cost_efficiency_pct,
    compute_jain_fairness,
    compute_over_provisioning_ratio,
    compute_reaction_time,
    compute_settling_time,
    compute_under_provisioning_ratio,
    effect_size_interpretation,
    mann_whitney_with_effect_size,
)


# ======================================================================
# Base metrics (metrics.py)
# ======================================================================

class TestSLACompliance:
    def test_perfect_compliance(self):
        assert compute_sla_compliance([50, 100, 150, 199]) == 100.0

    def test_zero_compliance(self):
        assert compute_sla_compliance([201, 300, 500]) == 0.0

    def test_partial_compliance(self):
        result = compute_sla_compliance([100, 200, 300])
        assert result == pytest.approx(66.666, abs=0.01)

    def test_boundary_at_target(self):
        assert compute_sla_compliance([200.0], 200.0) == 100.0

    def test_empty_list(self):
        assert compute_sla_compliance([]) == 100.0

    def test_custom_target(self):
        assert compute_sla_compliance([150], 100.0) == 0.0


class TestAverageCost:
    def test_normal(self, sample_costs):
        assert compute_average_cost(sample_costs) == pytest.approx(0.56)

    def test_empty(self):
        assert compute_average_cost([]) == 0.0

    def test_single(self):
        assert compute_average_cost([1.5]) == 1.5


class TestChurn:
    def test_normal(self, sample_replicas):
        # |3-2| + |5-3| + |5-5| + |7-5| + |7-7| + |5-7| + |4-5| + |3-4| + |2-3| = 1+2+0+2+0+2+1+1+1 = 10
        assert compute_churn(sample_replicas) == 10

    def test_stable(self):
        assert compute_churn([5, 5, 5, 5]) == 0

    def test_single_element(self):
        assert compute_churn([5]) == 0

    def test_empty(self):
        assert compute_churn([]) == 0


class TestMaxLatency:
    def test_normal(self, sample_latencies):
        assert compute_max_latency(sample_latencies) == 250.0

    def test_empty(self):
        assert compute_max_latency([]) == 0.0


class TestAverageLatency:
    def test_normal(self):
        assert compute_average_latency([100, 200, 300]) == pytest.approx(200.0)

    def test_empty(self):
        assert compute_average_latency([]) == 0.0


class TestPercentileLatency:
    def test_p99(self):
        lats = list(range(1, 101))  # 1 to 100
        assert compute_percentile_latency(lats, 99.0) == 100

    def test_p50(self):
        lats = list(range(1, 101))
        assert compute_percentile_latency(lats, 50.0) == 51

    def test_empty(self):
        assert compute_percentile_latency([], 95.0) == 0.0


class TestViolationSeverity:
    def test_with_violations(self, sample_latencies):
        result = compute_violation_severity(sample_latencies, 200.0)
        # Violations: 210-200=10, 250-200=50 → mean = 30
        assert result == pytest.approx(30.0)

    def test_no_violations(self):
        assert compute_violation_severity([50, 100, 150], 200.0) == 0.0

    def test_empty(self):
        assert compute_violation_severity([], 200.0) == 0.0


class TestMeanUtilization:
    def test_normal(self):
        assert compute_mean_utilization([0.5, 0.7, 0.9]) == pytest.approx(0.7)

    def test_empty(self):
        assert compute_mean_utilization([]) == 0.0


class TestTotalReward:
    def test_normal(self):
        assert compute_total_reward([1.0, -0.5, 0.3]) == pytest.approx(0.8)

    def test_empty(self):
        assert compute_total_reward([]) == 0.0


# ======================================================================
# Extended metrics (extended_metrics.py)
# ======================================================================

class TestCohensD:
    def test_identical_groups(self):
        assert cohens_d([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_large_effect(self):
        d = cohens_d([10, 11, 12, 13, 14], [1, 2, 3, 4, 5])
        assert abs(d) > 0.8  # large effect

    def test_insufficient_data(self):
        assert cohens_d([1], [2]) == 0.0
        assert cohens_d([], [1, 2]) == 0.0


class TestEffectSizeInterpretation:
    def test_negligible(self):
        assert effect_size_interpretation(0.1) == "negligible"

    def test_small(self):
        assert effect_size_interpretation(0.3) == "small"

    def test_medium(self):
        assert effect_size_interpretation(0.6) == "medium"

    def test_large(self):
        assert effect_size_interpretation(1.0) == "large"


class TestBootstrapCI:
    def test_known_data(self):
        data = [1.0] * 100
        point, lo, hi = bootstrap_ci(data)
        assert point == pytest.approx(1.0)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_single_value(self):
        point, lo, hi = bootstrap_ci([5.0])
        assert point == 5.0

    def test_ci_contains_mean(self):
        np.random.seed(42)
        data = np.random.normal(100, 10, 200).tolist()
        point, lo, hi = bootstrap_ci(data)
        assert lo < point < hi

    def test_empty(self):
        point, lo, hi = bootstrap_ci([])
        assert point == 0.0


class TestBonferroniCorrect:
    def test_single(self):
        assert bonferroni_correct([0.01]) == [0.01]

    def test_multiple(self):
        result = bonferroni_correct([0.01, 0.02, 0.05])
        assert result == pytest.approx([0.03, 0.06, 0.15])

    def test_clamping(self):
        result = bonferroni_correct([0.5, 0.5])
        assert result == [1.0, 1.0]

    def test_empty(self):
        assert bonferroni_correct([]) == []


class TestMannWhitneyWithEffectSize:
    def test_significant_difference(self):
        a = list(range(100, 120))
        b = list(range(1, 21))
        result = mann_whitney_with_effect_size(a, b)
        assert result["p_value"] < 0.05
        assert abs(result["cohens_d"]) > 0.8

    def test_insufficient_data(self):
        result = mann_whitney_with_effect_size([1], [2])
        assert result["p_value"] == 1.0


class TestReactionTime:
    def test_with_spike(self):
        rates = [100] * 10 + [200] + [200] * 9
        reps = [5] * 10 + [5, 5, 6] + [6] * 7
        rt = compute_reaction_time(rates, reps, spike_threshold=1.3)
        assert rt == pytest.approx(2.0)  # spike at 10, scale-up at 12

    def test_no_spike(self):
        rates = [100] * 20
        reps = [5] * 20
        assert math.isnan(compute_reaction_time(rates, reps))


class TestOverProvisioningRatio:
    def test_over(self):
        ratio = compute_over_provisioning_ratio([100], [10], per_pod_capacity=20.0)
        # ideal = ceil(100/20) = 5, actual = 10 → ratio = 2.0
        assert ratio == pytest.approx(2.0)

    def test_exact(self):
        ratio = compute_over_provisioning_ratio([100], [5], per_pod_capacity=20.0)
        assert ratio == pytest.approx(1.0)


class TestUnderProvisioningRatio:
    def test_none(self):
        assert compute_under_provisioning_ratio([0.5, 0.6, 0.7]) == 0.0

    def test_all(self):
        assert compute_under_provisioning_ratio([0.95, 0.99, 1.0]) == 1.0

    def test_partial(self):
        assert compute_under_provisioning_ratio([0.5, 0.95]) == 0.5


class TestSettlingTime:
    def test_quick_settle(self):
        lats = [150] * 5 + [250] + [150] * 20
        st = compute_settling_time(lats, sla_target=200.0)
        # Violation at step 5, back to compliant at step 6, needs 5 consecutive
        assert st == pytest.approx(1.0)

    def test_no_violation(self):
        assert math.isnan(compute_settling_time([100, 100, 100]))


class TestActionEntropy:
    def test_uniform(self):
        # All 7 actions equally used → max entropy
        actions = list(range(7)) * 100
        ent = compute_action_entropy(actions)
        assert ent == pytest.approx(math.log2(7), abs=0.01)

    def test_single_action(self):
        actions = [3] * 100
        assert compute_action_entropy(actions) == 0.0

    def test_empty(self):
        assert compute_action_entropy([]) == 0.0


class TestJainFairness:
    def test_uniform(self):
        assert compute_jain_fairness([5, 5, 5, 5]) == pytest.approx(1.0)

    def test_skewed(self):
        idx = compute_jain_fairness([10, 0, 0, 0])
        assert idx < 0.5

    def test_empty(self):
        assert compute_jain_fairness([]) == 1.0


class TestCoefficientOfVariation:
    def test_zero_variation(self):
        assert compute_coefficient_of_variation([5, 5, 5]) == 0.0

    def test_high_variation(self):
        cv = compute_coefficient_of_variation([1, 100, 1, 100])
        assert cv > 0.5

    def test_single_value(self):
        assert compute_coefficient_of_variation([5]) == 0.0


class TestCostEfficiency:
    def test_rl_cheaper(self):
        pct = compute_cost_efficiency_pct([0.40], [0.60])
        assert pct == pytest.approx(33.33, abs=0.1)

    def test_rl_same(self):
        assert compute_cost_efficiency_pct([0.50], [0.50]) == pytest.approx(0.0)

    def test_rl_expensive(self):
        pct = compute_cost_efficiency_pct([0.80], [0.40])
        assert pct < 0  # RL is worse
