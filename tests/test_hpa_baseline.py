"""Tests for RealisticHPA baseline."""

from __future__ import annotations

import pytest

from src.agents.hpa_baseline import RealisticHPA
from src.safety.safety_filter import ClusterState


class TestHPAFormula:
    def test_scaleup_on_high_cpu(self, hpa):
        """HPA should scale up when CPU is above target."""
        state = ClusterState(replicas=5, p99_latency=100.0, cpu_util=0.9)
        delta = hpa.act(state)
        assert delta > 0, "HPA should scale up on high CPU"

    def test_no_change_at_target(self, hpa):
        """HPA should not change when CPU is at target."""
        state = ClusterState(replicas=5, p99_latency=100.0, cpu_util=0.50)
        delta = hpa.act(state)
        assert delta == 0

    def test_scaledown_on_low_cpu(self):
        """HPA should eventually scale down on low CPU."""
        hpa = RealisticHPA(stabilization_window=1)  # short window for testing
        state = ClusterState(replicas=10, p99_latency=50.0, cpu_util=0.1)
        # Fill the window with low readings
        for _ in range(3):
            delta = hpa.act(state)
        # After filling window, should try to scale down
        delta = hpa.act(state)
        assert delta <= 0

    def test_respects_min_replicas(self, hpa):
        state = ClusterState(replicas=2, p99_latency=50.0, cpu_util=0.01)
        delta = hpa.act(state)
        assert state.replicas + delta >= hpa.min_replicas

    def test_respects_max_replicas(self, hpa):
        state = ClusterState(replicas=30, p99_latency=100.0, cpu_util=0.99)
        delta = hpa.act(state)
        assert state.replicas + delta <= hpa.max_replicas


class TestStabilizationWindow:
    def test_scaledown_delayed(self):
        """Scale-down should be delayed by the stabilization window."""
        hpa = RealisticHPA(stabilization_window=6)
        state_low = ClusterState(replicas=10, p99_latency=50.0, cpu_util=0.1)
        state_high = ClusterState(replicas=10, p99_latency=50.0, cpu_util=0.9)

        # First send a high reading → records desired=18 in window
        hpa.act(state_high)
        # Now send low readings — window still has the high value
        delta = hpa.act(state_low)
        # Because window max is still 18, scale-down won't happen
        assert delta >= 0

    def test_reset_clears_window(self, hpa):
        state = ClusterState(replicas=10, p99_latency=50.0, cpu_util=0.9)
        hpa.act(state)
        assert len(hpa.replica_history) > 0
        hpa.reset()
        assert len(hpa.replica_history) == 0


class TestHPAEdgeCases:
    def test_zero_cpu_handled(self, hpa):
        """CPU=0 should not cause division errors."""
        state = ClusterState(replicas=5, p99_latency=50.0, cpu_util=0.0)
        delta = hpa.act(state)
        # Should not crash, delta should be finite
        assert isinstance(delta, int)

    def test_repr(self, hpa):
        r = repr(hpa)
        assert "RealisticHPA" in r
        assert "target_cpu" in r
