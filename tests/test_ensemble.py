"""Tests for the Ensemble Meta-Agent and Workload Classifier."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.agents.ensemble_agent import (
    ClassifierThresholds,
    EnsembleMetaAgent,
    ROUTING_TABLE,
    WorkloadClassifier,
    WorkloadFeatures,
)


# ======================================================================
# WorkloadClassifier tests
# ======================================================================

class TestWorkloadClassifier:
    """Test the sliding-window workload classifier."""

    def _feed_window(self, classifier, rates):
        """Feed a list of request rates as observations."""
        for rate in rates:
            obs = np.zeros(23, dtype=np.float32)
            obs[4] = rate / 500.0  # request_rate / 500
            classifier.update(obs)
        return classifier.classification

    def test_uncertain_with_insufficient_data(self):
        """Should return 'uncertain' until window is full."""
        c = WorkloadClassifier(window_size=10)
        obs = np.zeros(23, dtype=np.float32)
        obs[4] = 100.0 / 500.0
        for _ in range(9):
            result = c.update(obs)
            assert result == "uncertain"

    def test_steady_pattern(self):
        """Low variance, no trend → steady."""
        c = WorkloadClassifier(window_size=10)
        # Flat 100 rps with tiny noise
        rates = [100 + np.random.normal(0, 2) for _ in range(10)]
        result = self._feed_window(c, rates)
        assert result == "steady"

    def test_noisy_pattern(self):
        """High CV → noisy."""
        c = WorkloadClassifier(window_size=10)
        # Wild swings: alternating 50 and 300
        rates = [50, 300, 40, 350, 60, 280, 45, 320, 55, 290]
        result = self._feed_window(c, rates)
        assert result == "noisy"

    def test_ramp_pattern(self):
        """Strong positive slope → ramp."""
        c = WorkloadClassifier(window_size=10)
        # Linear ramp from 50 to 100 in 10 steps
        rates = [50 + 5 * i for i in range(10)]  # slope ≈ 5
        result = self._feed_window(c, rates)
        assert result in ("ramp", "sawtooth")  # slope > 2.5

    def test_sawtooth_detection(self):
        """Very steep slope → sawtooth."""
        c = WorkloadClassifier(window_size=10)
        # Very steep ramp: slope = 10
        rates = [50 + 10 * i for i in range(10)]
        result = self._feed_window(c, rates)
        assert result == "sawtooth"

    def test_spike_detection(self):
        """Sudden large rate change → spike."""
        c = WorkloadClassifier(window_size=10)
        # Flat baseline with a single sudden jump to trigger spike_ratio > 0.20,
        # but keeps CV < 0.35 so it doesn't trigger 'noisy'.
        rates = [100, 100, 100, 100, 200, 100, 100, 100, 100, 100]
        result = self._feed_window(c, rates)
        assert result == "spike"

    def test_reset_clears_history(self):
        c = WorkloadClassifier(window_size=10)
        rates = [100] * 10
        self._feed_window(c, rates)
        assert c.classification != "uncertain"
        c.reset()
        assert c.classification == "uncertain"
        assert len(c._rate_history) == 0

    def test_confidence_in_range(self):
        c = WorkloadClassifier(window_size=10)
        rates = [100] * 10
        self._feed_window(c, rates)
        assert 0.0 <= c.confidence <= 1.0

    def test_features_populated(self):
        c = WorkloadClassifier(window_size=10)
        rates = [100 + i * 3 for i in range(10)]
        self._feed_window(c, rates)
        f = c.features
        assert f.mean_rate > 0
        assert f.slope > 0
        assert f.cv >= 0


class TestClassifierThresholds:
    """Test custom threshold overrides."""

    def test_custom_noisy_threshold(self):
        # Set very low noisy threshold → even mild variance triggers noisy
        t = ClassifierThresholds(noisy_cv=0.05)
        c = WorkloadClassifier(window_size=5, thresholds=t)
        obs = np.zeros(23, dtype=np.float32)
        # Feed rates with CV ≈ 0.1 (above 0.05 custom threshold)
        rates = [90, 110, 95, 105, 100]
        for rate in rates:
            obs[4] = rate / 500.0
            c.update(obs)
        assert c.classification == "noisy"

    def test_custom_ramp_threshold(self):
        t = ClassifierThresholds(ramp_slope=1.0)  # very sensitive
        c = WorkloadClassifier(window_size=5, thresholds=t)
        obs = np.zeros(23, dtype=np.float32)
        rates = [100, 102, 104, 106, 108]  # slope = 2
        for rate in rates:
            obs[4] = rate / 500.0
            c.update(obs)
        assert c.classification in ("ramp", "sawtooth")


class TestWorkloadFeatures:
    """Test feature extraction."""

    def test_constant_rates(self):
        c = WorkloadClassifier(window_size=5)
        obs = np.zeros(23, dtype=np.float32)
        for _ in range(5):
            obs[4] = 100.0 / 500.0
            c.update(obs)
        f = c.features
        assert f.cv == pytest.approx(0.0)
        assert f.slope == pytest.approx(0.0, abs=0.01)
        assert f.delta_variance == pytest.approx(0.0)
        assert f.spike_ratio == pytest.approx(0.0)
        assert f.mean_rate == pytest.approx(100.0)


# ======================================================================
# Routing table tests
# ======================================================================

class TestRoutingTable:
    def test_all_classifications_have_routes(self):
        expected = {"noisy", "ramp", "sawtooth", "spike", "steady", "uncertain"}
        assert set(ROUTING_TABLE.keys()) == expected

    def test_noisy_routes_to_qrdqn(self):
        assert ROUTING_TABLE["noisy"] == "qrdqn"

    def test_ramp_routes_to_dqn(self):
        assert ROUTING_TABLE["ramp"] == "dqn"

    def test_sawtooth_routes_to_dqn(self):
        assert ROUTING_TABLE["sawtooth"] == "dqn"

    def test_spike_routes_to_ppo_1m(self):
        assert ROUTING_TABLE["spike"] == "ppo_1m"

    def test_steady_routes_to_ppo_1m(self):
        assert ROUTING_TABLE["steady"] == "ppo_1m"

    def test_uncertain_routes_to_fallback(self):
        assert ROUTING_TABLE["uncertain"] == "ppo"


# ======================================================================
# Integration-level tests (only run if models are available)
# ======================================================================

class TestEnsembleIntegration:
    """Integration tests — skip if models aren't downloaded."""

    @pytest.fixture
    def ensemble(self):
        try:
            return EnsembleMetaAgent()
        except (ValueError, Exception):
            pytest.skip("Specialist models not available for integration test")

    def test_decide_returns_valid_delta(self, ensemble):
        obs = np.zeros(23, dtype=np.float32)
        obs[4] = 100.0 / 500.0
        delta = ensemble.decide(obs, step=0)
        assert -3 <= delta <= 3

    def test_decide_with_info_has_keys(self, ensemble):
        obs = np.zeros(23, dtype=np.float32)
        obs[4] = 100.0 / 500.0
        # Feed enough observations for classification
        for i in range(10):
            info = ensemble.decide_with_info(obs, step=i)
        assert "pattern" in info
        assert "specialist" in info
        assert "confidence" in info
        assert "delta" in info

    def test_routing_summary(self, ensemble):
        obs = np.zeros(23, dtype=np.float32)
        obs[4] = 100.0 / 500.0
        for i in range(15):
            ensemble.decide(obs, step=i)
        summary = ensemble.routing_summary
        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_reset(self, ensemble):
        obs = np.zeros(23, dtype=np.float32)
        obs[4] = 100.0 / 500.0
        ensemble.decide(obs, step=0)
        ensemble.reset()
        assert ensemble.classifier.classification == "uncertain"

    def test_repr(self, ensemble):
        r = repr(ensemble)
        assert "EnsembleMetaAgent" in r
        assert "specialists" in r
