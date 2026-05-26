"""
K8sSimEnv — Gymnasium environment that simulates a Kubernetes cluster.

Models pod cold-start delay, request queue buildup, multi-objective cost,
and 5 traffic patterns.  Designed for RecurrentPPO training at 1000+ steps/sec.

MDP
---
State   : 22-dimensional float vector (see _build_obs)
Action  : Discrete(7) → replica delta in {-3, -2, -1, 0, +1, +2, +3}
Step    : 30 simulated seconds
Episode : 120 steps = 1 simulated hour
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from src.env.workload import WorkloadGenerator

# Default config path (can be overridden)
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "env_config.yaml"


class K8sSimEnv(gym.Env):
    """Kubernetes cluster simulator for RL autoscaler training.

    Parameters
    ----------
    config_path : str | Path | None
        Path to env_config.yaml.  ``None`` uses the default.
    seed : int | None
        RNG seed for reproducibility.
    workload_pattern : str
        Specific workload pattern, or ``"random"`` for domain randomization.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str | Path | None = None,
        seed: int | None = None,
        workload_pattern: str = "random",
    ) -> None:
        super().__init__()

        # ── Load configuration ───────────────────────────────────────
        cfg_path = Path(config_path) if config_path else _DEFAULT_CONFIG
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)["simulator"]

        # ── Spaces ───────────────────────────────────────────────────
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=-1.0, high=np.inf, shape=(22,), dtype=np.float32,
        )

        # ── Fixed parameters from config ─────────────────────────────
        self.episode_length: int = self.cfg["episode_length"]
        self.step_duration: int = self.cfg["step_duration"]
        self.sla_target: float = float(self.cfg["sla_target"])
        self.base_latency: float = float(self.cfg["base_latency"])
        self.min_replicas: int = self.cfg["min_replicas"]
        self.max_replicas: int = self.cfg["max_replicas"]
        self.num_nodes: int = self.cfg["num_nodes"]
        self.node_cpu: float = float(self.cfg["node_cpu"])
        self.node_mem: float = float(self.cfg["node_mem"])
        self.cpu_per_pod: float = float(self.cfg["cpu_per_pod"])
        self.mem_per_pod: float = float(self.cfg["mem_per_pod"])
        self.packing: float = float(self.cfg["node_packing_factor"])

        # Randomization ranges
        rand = self.cfg["randomization"]
        self._ppc_range = (rand["per_pod_capacity_min"], rand["per_pod_capacity_max"])
        self._csm_range = (rand["cold_start_mean_min"], rand["cold_start_mean_max"])
        self._cs_sigma: float = float(rand["cold_start_sigma"])
        self._cs_clip = (rand["cold_start_clip_min"], rand["cold_start_clip_max"])
        self._np_range = (rand["node_price_min"], rand["node_price_max"])

        # Reward weights
        rw = self.cfg["reward"]
        self._w_sla: float = float(rw["sla_weight"])
        self._w_cost: float = float(rw["cost_weight"])
        self._w_crash: float = float(rw["crash_multiplier"])
        self._crash_thresh: float = float(rw["crash_threshold"])
        self._w_stab: float = float(rw["stability_weight"])
        self._stab_thresh: float = float(rw["stability_threshold"])
        cap = rw.get("sla_violation_cap")
        self._sla_violation_cap: float | None = float(cap) if cap is not None else None

        # ── RNG & workload ───────────────────────────────────────────
        self._rng = np.random.default_rng(seed)
        self._workload_pattern = workload_pattern
        self.workload = WorkloadGenerator(pattern=workload_pattern, seed=seed)

        # ── Mutable state (set in reset) ─────────────────────────────
        self.replicas: int = 0
        self.pending_pods: list[float] = []  # remaining startup time per pod
        self.queue_depth: float = 0.0
        self.request_rate: float = 0.0
        self.prev_request_rate: float = 0.0
        self.p99_latency: float = 0.0
        self.cpu_util: float = 0.0
        self.prev_cpu_util: float = 0.0
        self.mem_util: float = 0.0
        self.cost_rate: float = 0.0
        self.step_count: int = 0

        # Domain-randomized per episode
        self.per_pod_capacity: float = 0.0
        self.cold_start_mean: float = 0.0
        self.node_price: float = 0.0

    # ==================================================================
    # Gymnasium interface
    # ==================================================================

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment for a new episode with domain randomization."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # ── Domain randomization ─────────────────────────────────────
        self.per_pod_capacity = float(
            self._rng.uniform(*self._ppc_range)
        )
        self.cold_start_mean = float(
            self._rng.uniform(*self._csm_range)
        )
        self.node_price = float(
            self._rng.uniform(*self._np_range)
        )

        # ── Initial cluster state ────────────────────────────────────
        self.replicas = self.min_replicas  # start at minimum
        self.pending_pods = []
        self.queue_depth = 0.0
        self.step_count = 0

        # Workload
        self.workload.reset()
        self.request_rate = self.workload.get_rate(0)
        self.prev_request_rate = self.request_rate

        # Derived metrics at initial state
        self.cpu_util = self._compute_cpu_util()
        self.prev_cpu_util = self.cpu_util
        self.mem_util = self._compute_mem_util()
        self.p99_latency = self.base_latency
        self.cost_rate = self._compute_cost()

        obs = self._build_obs()
        info = self._build_info(delta=0)
        return obs, info

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one simulation step (30 simulated seconds).

        Parameters
        ----------
        action : int
            Integer 0–6 mapping to replica delta -3 to +3.
        """
        delta = int(action) - 3  # map [0..6] → [-3..+3]

        # 1. Apply scaling action
        self._apply_scaling(delta)

        # 2. Age pending pods → graduate to ready
        self._age_pending_pods()

        # 3. Update traffic
        self.step_count += 1
        self.prev_request_rate = self.request_rate
        self.request_rate = self.workload.get_rate(self.step_count)

        # 4. Queue dynamics
        self._update_queue()

        # 5. Compute utilization
        self.prev_cpu_util = self.cpu_util
        self.cpu_util = self._compute_cpu_util()
        self.mem_util = self._compute_mem_util()

        # 6. Compute cost
        self.cost_rate = self._compute_cost()

        # 7. Compute reward
        reward = self._compute_reward(delta)

        # 8. Build observation
        obs = self._build_obs()

        # 9. Check termination
        terminated = False
        truncated = self.step_count >= self.episode_length

        info = self._build_info(delta)
        return obs, reward, terminated, truncated, info

    # ==================================================================
    # Scaling logic
    # ==================================================================

    def _apply_scaling(self, delta: int) -> None:
        """Apply the replica delta, creating pending pods or removing ready ones."""
        if delta > 0:
            # Scale up — create pending pods
            for _ in range(delta):
                total = self.replicas + len(self.pending_pods)
                if total >= self.max_replicas:
                    break
                # Startup time from LogNormal, clipped
                raw = float(self._rng.lognormal(
                    mean=np.log(self.cold_start_mean), sigma=self._cs_sigma,
                ))
                startup = np.clip(raw, *self._cs_clip)
                self.pending_pods.append(startup)

        elif delta < 0:
            # Scale down — cancel pending first (cheaper), then remove ready
            to_remove = abs(delta)
            # Cancel pending pods (newest first = most time remaining)
            while to_remove > 0 and self.pending_pods:
                self.pending_pods.pop()
                to_remove -= 1
            # Remove ready pods
            while to_remove > 0 and self.replicas > self.min_replicas:
                self.replicas -= 1
                to_remove -= 1

    def _age_pending_pods(self) -> None:
        """Decrease pending pod timers by one step; graduate those that reach 0."""
        still_pending: list[float] = []
        for remaining in self.pending_pods:
            remaining -= self.step_duration
            if remaining <= 0:
                # Pod graduated to ready
                if self.replicas < self.max_replicas:
                    self.replicas += 1
            else:
                still_pending.append(remaining)
        self.pending_pods = still_pending

    # ==================================================================
    # Queue dynamics (Little's Law approximation)
    # ==================================================================

    def _update_queue(self) -> None:
        """Compute queue depth and P99 latency from arrivals vs. service rate."""
        service_rate = self.replicas * self.per_pod_capacity
        arrivals = self.request_rate * self.step_duration
        departures = service_rate * self.step_duration

        self.queue_depth = max(0.0, self.queue_depth + arrivals - departures)

        # Queue wait time in ms
        if self.request_rate > 0:
            queue_wait = (self.queue_depth / self.request_rate) * 1000.0
        else:
            queue_wait = 0.0

        noise = self._rng.normal(0.0, 2.0)
        self.p99_latency = max(0.0, self.base_latency + queue_wait + noise)

    # ==================================================================
    # Utilization metrics
    # ==================================================================

    def _compute_cpu_util(self) -> float:
        """CPU utilization as fraction of total capacity used by requests."""
        if self.replicas == 0:
            return 1.0
        total_capacity = self.replicas * self.per_pod_capacity
        if total_capacity == 0:
            return 1.0
        return min(1.0, self.request_rate / total_capacity)

    def _compute_mem_util(self) -> float:
        """Memory utilization (simplified — correlated with CPU + noise)."""
        base = self.cpu_util * 0.7  # memory tracks CPU loosely
        noise = self._rng.normal(0.0, 0.05)
        return float(np.clip(base + noise, 0.0, 1.0))

    # ==================================================================
    # Cost model
    # ==================================================================

    def _compute_cost(self) -> float:
        """Compute $/hour based on nodes needed for current pod count."""
        total_pods = self.replicas + len(self.pending_pods)
        total_pod_cpu = total_pods * self.cpu_per_pod
        nodes_needed = math.ceil(total_pod_cpu / (self.node_cpu * self.packing))
        nodes_needed = max(1, nodes_needed)
        return nodes_needed * self.node_price

    # ==================================================================
    # Reward
    # ==================================================================

    def _compute_reward(self, delta: int) -> float:
        """Multi-objective reward: SLA + cost + crash + stability."""
        # SLA penalty — proportional to latency overshoot
        sla_violation = max(0.0, self.p99_latency - self.sla_target) / self.sla_target
        if self._sla_violation_cap is not None:
            sla_violation = min(sla_violation, self._sla_violation_cap)
        r_sla = self._w_sla * sla_violation

        # Cost penalty
        r_cost = self._w_cost * self.cost_rate

        # Crash penalty — catastrophic latency
        r_crash = self._w_crash if self.p99_latency > self._crash_thresh * self.sla_target else 0.0

        # Stability penalty — only when latency is comfortable
        if self.p99_latency < self._stab_thresh * self.sla_target:
            r_stability = self._w_stab * abs(delta)
        else:
            r_stability = 0.0

        return r_sla + r_cost + r_crash + r_stability

    # ==================================================================
    # Observation vector (22 dimensions)
    # ==================================================================

    def _build_obs(self) -> np.ndarray:
        """Build the 22-dimensional normalized observation vector.

        Layout
        ------
        [0]  cpu_util
        [1]  mem_util
        [2]  replicas / max_replicas
        [3]  len(pending_pods) / 10
        [4]  request_rate / 500
        [5]  (request_rate - prev_rate) / 100
        [6]  p99_latency / 1000
        [7]  per_pod_capacity / 30
        [8]  queue_depth / 10000
        [9-17]  per-node: cpu_avail, mem_avail, pod_count  (×3 nodes)
        [18] sin(2π × step / 120)
        [19] cos(2π × step / 120)
        [20] cost_rate / 2.0
        [21] prev_cpu_util
        """
        obs = np.zeros(22, dtype=np.float32)

        # Per-deployment (9 dims)
        obs[0] = self.cpu_util
        obs[1] = self.mem_util
        obs[2] = self.replicas / self.max_replicas
        obs[3] = len(self.pending_pods) / 10.0
        obs[4] = self.request_rate / 500.0
        obs[5] = (self.request_rate - self.prev_request_rate) / 100.0
        obs[6] = self.p99_latency / 1000.0
        obs[7] = self.per_pod_capacity / 30.0
        obs[8] = self.queue_depth / 10000.0

        # Per-node (3 nodes × 3 dims = 9 dims)
        total_pods = self.replicas + len(self.pending_pods)
        pods_per_node = total_pods / self.num_nodes if self.num_nodes > 0 else 0
        cpu_used_per_node = pods_per_node * self.cpu_per_pod
        mem_used_per_node = pods_per_node * self.mem_per_pod

        for i in range(self.num_nodes):
            base = 9 + i * 3
            obs[base] = max(0.0, (self.node_cpu - cpu_used_per_node) / self.node_cpu)
            obs[base + 1] = max(0.0, (self.node_mem - mem_used_per_node) / self.node_mem)
            obs[base + 2] = pods_per_node / 30.0

        # Global (4 dims)
        phase = 2.0 * np.pi * self.step_count / self.episode_length
        obs[18] = np.sin(phase)
        obs[19] = np.cos(phase)
        obs[20] = self.cost_rate / 2.0
        obs[21] = self.prev_cpu_util

        return obs

    # ==================================================================
    # Info dict
    # ==================================================================

    def _build_info(self, delta: int) -> dict[str, Any]:
        """Build the info dictionary returned with each step."""
        return {
            "step": self.step_count,
            "replicas": self.replicas,
            "pending_pods": len(self.pending_pods),
            "request_rate": self.request_rate,
            "p99_latency": self.p99_latency,
            "cpu_util": self.cpu_util,
            "mem_util": self.mem_util,
            "queue_depth": self.queue_depth,
            "cost_rate": self.cost_rate,
            "delta_applied": delta,
            "sla_breach": self.p99_latency > self.sla_target,
            "workload_pattern": self.workload.pattern,
        }

    def __repr__(self) -> str:
        return (
            f"K8sSimEnv(step={self.step_count}, replicas={self.replicas}, "
            f"pending={len(self.pending_pods)}, rps={self.request_rate:.0f}, "
            f"p99={self.p99_latency:.1f}ms)"
        )
