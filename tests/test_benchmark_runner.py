"""Smoke tests for benchmark_runner and baselines."""

from __future__ import annotations

import pytest
import numpy as np

from src.env.k8s_sim import K8sSimEnv
from src.evaluation.baselines import (
    FixedReplicaBaseline, OracleScaler, ReactiveThresholdBaseline,
)
from src.evaluation.benchmark_runner import run_episode_extended
from src.safety.safety_filter import ClusterState


class TestFixedReplicaBaseline:
    def test_always_returns_delta_to_target(self):
        b = FixedReplicaBaseline(fixed_replicas=7)
        state = ClusterState(replicas=3, p99_latency=50.0)
        assert b.act(state) == 4  # 7 - 3

    def test_at_target(self):
        b = FixedReplicaBaseline(fixed_replicas=5)
        state = ClusterState(replicas=5, p99_latency=50.0)
        assert b.act(state) == 0

    def test_scale_down(self):
        b = FixedReplicaBaseline(fixed_replicas=3)
        state = ClusterState(replicas=10, p99_latency=50.0)
        assert b.act(state) == -7


class TestReactiveThresholdBaseline:
    def test_scaleup_on_high_cpu(self):
        b = ReactiveThresholdBaseline(high_threshold=0.8, scale_up_step=2)
        state = ClusterState(replicas=5, p99_latency=50.0, cpu_util=0.9)
        assert b.act(state) == 2

    def test_scaledown_on_low_cpu(self):
        b = ReactiveThresholdBaseline(low_threshold=0.3, scale_down_step=1)
        state = ClusterState(replicas=5, p99_latency=50.0, cpu_util=0.1)
        assert b.act(state) == -1

    def test_no_action_in_range(self):
        b = ReactiveThresholdBaseline()
        state = ClusterState(replicas=5, p99_latency=50.0, cpu_util=0.5)
        assert b.act(state) == 0

    def test_respects_min_replicas(self):
        b = ReactiveThresholdBaseline(min_replicas=2)
        state = ClusterState(replicas=2, p99_latency=50.0, cpu_util=0.1)
        delta = b.act(state)
        assert state.replicas + delta >= 2


class TestOracleScaler:
    def test_follows_plan(self):
        rates = [100.0] * 120
        oracle = OracleScaler(future_rates=rates, per_pod_capacity=20.0)
        # ideal = ceil(100/20) = 5
        state = ClusterState(replicas=2, p99_latency=50.0)
        delta = oracle.act(state)
        assert delta == 3  # clipped to ±3 (target 5, current 2)

    def test_delta_clamped(self):
        rates = [500.0] * 120
        oracle = OracleScaler(future_rates=rates, per_pod_capacity=20.0)
        state = ClusterState(replicas=2, p99_latency=50.0)
        delta = oracle.act(state)
        assert -3 <= delta <= 3

    def test_reset(self):
        rates = [100.0] * 120
        oracle = OracleScaler(future_rates=rates)
        state = ClusterState(replicas=5, p99_latency=50.0)
        oracle.act(state)
        oracle.act(state)
        oracle.reset()
        # After reset, step counter should be back to 0
        assert oracle._step == 0


class TestRunEpisodeExtended:
    def test_smoke_with_hpa(self):
        """Run a full episode with HPA and check all expected keys."""
        from src.agents.hpa_baseline import RealisticHPA
        env = K8sSimEnv(workload_pattern="steady", seed=42)
        env.reset(seed=42)
        hpa = RealisticHPA()
        result = run_episode_extended(env, hpa, agent_type="hpa")

        expected_keys = [
            "sla_compliance", "avg_cost", "churn", "max_latency",
            "avg_latency", "p95_latency", "p99_latency", "severity",
            "cpu_util", "reward", "pattern", "reaction_time",
            "over_provisioning", "under_provisioning", "settling_time",
            "action_entropy", "replica_fairness", "composite",
            "_trajectories",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        assert 0 <= result["sla_compliance"] <= 100
        assert result["avg_cost"] > 0
        assert result["composite"] >= 0

    def test_smoke_with_reactive(self):
        env = K8sSimEnv(workload_pattern="diurnal", seed=42)
        env.reset(seed=42)
        reactive = ReactiveThresholdBaseline()
        result = run_episode_extended(env, reactive, agent_type="reactive")
        assert "sla_compliance" in result
        assert result["pattern"] == "diurnal"

    def test_smoke_with_fixed(self):
        env = K8sSimEnv(workload_pattern="flash_crowd", seed=42)
        env.reset(seed=42)
        fixed = FixedReplicaBaseline(fixed_replicas=8)
        result = run_episode_extended(env, fixed, agent_type="fixed")
        assert result["avg_cost"] > 0
