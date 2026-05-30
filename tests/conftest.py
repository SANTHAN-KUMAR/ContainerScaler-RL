"""Shared pytest fixtures for ContainerScaler-RL tests."""

from __future__ import annotations

import pytest
import numpy as np

from src.env.k8s_sim import K8sSimEnv
from src.env.workload import WorkloadGenerator
from src.agents.hpa_baseline import RealisticHPA
from src.safety.safety_filter import SafetyFilter, ClusterState


@pytest.fixture
def env():
    """Create a standard K8sSimEnv with fixed seed."""
    e = K8sSimEnv(workload_pattern="steady", seed=42)
    e.reset(seed=42)
    return e


@pytest.fixture
def env_random():
    """Create a K8sSimEnv with random workload pattern."""
    e = K8sSimEnv(workload_pattern="random", seed=123)
    e.reset(seed=123)
    return e


@pytest.fixture
def hpa():
    """Create a default RealisticHPA instance."""
    return RealisticHPA()


@pytest.fixture
def safety():
    """Create a default SafetyFilter instance."""
    return SafetyFilter()


@pytest.fixture
def sample_obs(env):
    """Get a sample observation from the environment."""
    obs, _ = env.reset(seed=42)
    return obs


@pytest.fixture
def sample_latencies():
    """Sample latency data for metric testing."""
    return [50.0, 80.0, 120.0, 190.0, 210.0, 250.0, 100.0, 150.0, 180.0, 195.0]


@pytest.fixture
def sample_costs():
    """Sample cost data for metric testing."""
    return [0.40, 0.40, 0.60, 0.60, 0.80, 0.80, 0.60, 0.60, 0.40, 0.40]


@pytest.fixture
def sample_replicas():
    """Sample replica trajectory."""
    return [2, 3, 5, 5, 7, 7, 5, 4, 3, 2]
