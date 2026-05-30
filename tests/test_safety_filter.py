"""Tests for SafetyFilter."""

from __future__ import annotations

import pytest
import numpy as np

from src.safety.safety_filter import ClusterState, SafetyFilter


class TestSafetyFilterRules:
    """Test each of the 4 safety rules independently."""

    def test_rule1_min_replicas(self, safety):
        """Cannot scale below min_replicas (2)."""
        state = ClusterState(replicas=2, p99_latency=50.0)
        delta = safety.check(state, proposed_delta=-3, step=10)
        assert state.replicas + delta >= safety.MIN_REPLICAS

    def test_rule1_max_replicas(self, safety):
        """Cannot scale above max_replicas (30)."""
        state = ClusterState(replicas=29, p99_latency=50.0)
        delta = safety.check(state, proposed_delta=3, step=10)
        assert state.replicas + delta <= safety.MAX_REPLICAS

    def test_rule2_no_scaledown_high_latency(self, safety):
        """No scale-down when p99 > 0.8 × SLA target."""
        state = ClusterState(replicas=10, p99_latency=170.0)  # > 0.8 × 200
        delta = safety.check(state, proposed_delta=-2, step=10)
        assert delta >= 0  # should not scale down

    def test_rule2_allows_scaledown_low_latency(self, safety):
        """Scale-down is fine when latency is comfortable."""
        state = ClusterState(replicas=10, p99_latency=50.0)
        delta = safety.check(state, proposed_delta=-2, step=10)
        assert delta == -2

    def test_rule3_max_delta_clamp(self, safety):
        """Delta is clamped to ±3."""
        state = ClusterState(replicas=10, p99_latency=50.0)
        # Even if we propose ±5, it should be clamped
        delta_up = safety.check(state, proposed_delta=5, step=10)
        assert delta_up <= 3
        delta_down = safety.check(state, proposed_delta=-5, step=20)
        assert delta_down >= -3

    def test_rule4_rate_limiting(self, safety):
        """Minimum 1 step between non-zero actions."""
        state = ClusterState(replicas=10, p99_latency=50.0)
        # First action goes through
        d1 = safety.check(state, proposed_delta=1, step=0)
        assert d1 == 1
        # Immediate second action should be blocked
        d2 = safety.check(state, proposed_delta=1, step=0)
        assert d2 == 0

    def test_rate_limiting_expires(self, safety):
        """After cooldown, actions go through again."""
        state = ClusterState(replicas=10, p99_latency=50.0)
        safety.check(state, proposed_delta=1, step=0)
        d = safety.check(state, proposed_delta=1, step=1)
        assert d == 1


class TestSafetyFilterEdgeCases:
    def test_zero_delta_always_passes(self, safety):
        state = ClusterState(replicas=10, p99_latency=250.0)
        delta = safety.check(state, proposed_delta=0, step=0)
        assert delta == 0

    def test_at_min_replicas_scaledown(self, safety):
        state = ClusterState(replicas=2, p99_latency=50.0)
        delta = safety.check(state, proposed_delta=-1, step=10)
        assert state.replicas + delta >= 2

    def test_at_max_replicas_scaleup(self, safety):
        state = ClusterState(replicas=30, p99_latency=50.0)
        delta = safety.check(state, proposed_delta=3, step=10)
        assert delta == 0

    def test_reset(self, safety):
        state = ClusterState(replicas=10, p99_latency=50.0)
        safety.check(state, proposed_delta=1, step=0)
        safety.reset()
        # After reset, action should go through even at step 0
        d = safety.check(state, proposed_delta=1, step=0)
        assert d == 1


class TestClusterStateFromObs:
    def test_round_trip(self):
        """from_obs should approximately recover known values."""
        # Build a known observation
        obs = np.zeros(23, dtype=np.float32)
        obs[0] = 0.5   # cpu_util
        obs[1] = 0.3   # mem_util
        obs[2] = 10.0 / 30.0  # replicas / max_replicas → 10
        obs[3] = 2.0 / 10.0   # pending_pods / 10 → 2
        obs[4] = 200.0 / 500.0  # request_rate / 500 → 200
        obs[6] = 150.0 / 1000.0  # p99_latency / 1000 → 150
        obs[8] = 500.0 / 10000.0  # queue_depth / 10000 → 500
        obs[20] = 0.80 / 2.0  # cost_rate / 2.0 → 0.80

        state = ClusterState.from_obs(obs)

        assert state.replicas == 10
        assert state.p99_latency == pytest.approx(150.0)
        assert state.cpu_util == pytest.approx(0.5)
        assert state.request_rate == pytest.approx(200.0)
        assert state.pending_pods == 2
        assert state.queue_depth == pytest.approx(500.0)
        assert state.mem_util == pytest.approx(0.3)
        assert state.cost_rate == pytest.approx(0.80)

    def test_from_obs_with_real_env(self, env, sample_obs):
        """from_obs should not crash on real observations."""
        state = ClusterState.from_obs(sample_obs)
        assert state.replicas >= 0
        assert state.p99_latency >= 0
