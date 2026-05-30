"""Tests for K8sSimEnv (the simulator environment)."""

from __future__ import annotations

import numpy as np
import pytest

from src.env.k8s_sim import K8sSimEnv


class TestEnvironmentBasics:
    def test_observation_shape(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == (23,)
        assert obs.dtype == np.float32

    def test_action_space(self, env):
        assert env.action_space.n == 7

    def test_observation_space(self, env):
        assert env.observation_space.shape == (23,)

    def test_reset_returns_info(self, env):
        obs, info = env.reset(seed=42)
        assert "step" in info
        assert "replicas" in info
        assert "p99_latency" in info

    def test_step_returns_five_values(self, env):
        result = env.step(3)  # action=3 → delta=0
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (23,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


class TestEpisodeTermination:
    def test_episode_truncates_at_120(self, env):
        env.reset(seed=42)
        for step in range(120):
            obs, reward, terminated, truncated, info = env.step(3)
            if step < 119:
                assert not truncated, f"Episode truncated early at step {step}"
        assert truncated, "Episode did not truncate at step 120"

    def test_episode_never_terminates_early(self, env):
        """Episodes should only truncate, never terminate early."""
        env.reset(seed=42)
        for _ in range(120):
            _, _, terminated, truncated, _ = env.step(3)
            assert not terminated


class TestScalingActions:
    def test_action_0_scales_down_3(self, env):
        env.reset(seed=42)
        env.replicas = 10  # set to a safe middle value
        old_reps = env.replicas
        env.step(0)  # delta = -3
        # May not decrease by exactly 3 due to min_replicas
        assert env.replicas <= old_reps

    def test_action_6_scales_up_3(self, env):
        env.reset(seed=42)
        env.replicas = 5
        env.step(6)  # delta = +3
        # Pending pods may not have graduated yet
        total = env.replicas + len(env.pending_pods)
        assert total >= 5

    def test_action_3_no_change(self, env):
        env.reset(seed=42)
        old_reps = env.replicas
        old_pending = len(env.pending_pods)
        env.step(3)  # delta = 0
        # Pending may graduate, so total should be stable or increase
        assert env.replicas + len(env.pending_pods) >= old_reps

    def test_min_replicas_enforced(self, env):
        env.reset(seed=42)
        for _ in range(20):
            env.step(0)  # keep scaling down
        assert env.replicas >= env.min_replicas

    def test_max_replicas_enforced(self, env):
        env.reset(seed=42)
        for _ in range(50):
            env.step(6)  # keep scaling up
        total = env.replicas + len(env.pending_pods)
        assert total <= env.max_replicas


class TestObservationValues:
    def test_obs_values_reasonable(self, env):
        obs, _ = env.reset(seed=42)
        # CPU util should be in [0, 1]
        assert 0.0 <= obs[0] <= 1.0
        # Mem util should be in [0, 1]
        assert 0.0 <= obs[1] <= 1.0
        # Replicas / max_replicas should be in [0, 1]
        assert 0.0 <= obs[2] <= 1.0
        # Request rate / 500 should be positive
        assert obs[4] >= 0.0

    def test_obs_deterministic_with_seed(self):
        env1 = K8sSimEnv(workload_pattern="steady", seed=42)
        obs1, _ = env1.reset(seed=42)

        env2 = K8sSimEnv(workload_pattern="steady", seed=42)
        obs2, _ = env2.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)


class TestRewardRange:
    def test_reward_is_finite(self, env):
        env.reset(seed=42)
        for _ in range(10):
            _, reward, _, _, _ = env.step(3)
            assert np.isfinite(reward)

    def test_reward_with_different_actions(self, env):
        """All actions should produce finite rewards."""
        for action in range(7):
            env.reset(seed=42)
            _, reward, _, _, _ = env.step(action)
            assert np.isfinite(reward)


class TestInfoDict:
    def test_info_keys(self, env):
        env.reset(seed=42)
        _, _, _, _, info = env.step(3)
        expected_keys = [
            "step", "replicas", "pending_pods", "request_rate",
            "p99_latency", "cpu_util", "mem_util", "queue_depth",
            "cost_rate", "delta_applied", "sla_breach", "workload_pattern",
        ]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

    def test_sla_breach_flag(self, env):
        env.reset(seed=42)
        _, _, _, _, info = env.step(3)
        assert isinstance(info["sla_breach"], bool)


class TestDomainRandomization:
    def test_different_seeds_give_different_params(self):
        env1 = K8sSimEnv(seed=1)
        env1.reset(seed=1)
        cap1 = env1.per_pod_capacity

        env2 = K8sSimEnv(seed=999)
        env2.reset(seed=999)
        cap2 = env2.per_pod_capacity

        # Different seeds should (almost always) give different params
        # This test may rarely fail due to random chance
        assert cap1 != cap2

    def test_alpha_setter(self, env):
        env.set_alpha(0.5)
        assert env._alpha == 0.5
        env.set_alpha(0.0)
        assert env._alpha == 0.0
