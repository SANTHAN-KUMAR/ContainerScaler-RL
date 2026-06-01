"""
PrometheusObserver — The Sim-to-Real bridge.

Queries real cluster metrics and normalises them into the exact same
23-dimensional observation vector the agent saw during simulation training.

Target deployment: Online Boutique `frontend` (replaces the old podinfo default).
"""

from __future__ import annotations

import logging
import math

import numpy as np
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect

logger = logging.getLogger(__name__)


class PrometheusObserver:
    """Collects real cluster metrics and formats them for the RL agent.

    Parameters
    ----------
    prom_url : str
        Prometheus HTTP endpoint URL.
    namespace : str
        Kubernetes namespace (e.g., "default").
    deployment : str
        Target deployment name (e.g., "frontend").
    """

    def __init__(
        self,
        prom_url: str = "http://localhost:9090",
        namespace: str = "default",
        deployment: str = "frontend",
    ) -> None:
        self.prom = PrometheusConnect(url=prom_url, disable_ssl=True)
        self.namespace = namespace
        self.deployment = deployment

        try:
            config.load_kube_config()
            # Disable SSL verification for local K3s self-signed certs
            client.Configuration._default.verify_ssl = False
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            self.apps_v1 = client.AppsV1Api()
        except Exception as e:
            logger.warning(f"Could not load kubeconfig: {e}")
            self.apps_v1 = None

        # Match simulator config
        self.max_replicas = 30
        self.node_cpu = 4.0
        self.node_mem = 16.0
        self.num_nodes = 3

        # Stateful metrics for derivatives
        self.prev_request_rate: float = 0.0
        self.prev_prev_request_rate: float = 0.0
        self.prev_cpu_util: float = 0.0
        self.step_count: int = 0

    def get_state(self) -> np.ndarray:
        """Query Prometheus + K8s API and build the 23-dim normalised observation vector.

        Returns
        -------
        np.ndarray
            Observation vector matching simulator format exactly.
        """
        # ── 1. HTTP traffic from Traefik Ingress ─────────────────────────────
        # The frontend container in v0.10.5 does not natively expose HTTP metrics.
        # Locust traffic is routed through Traefik, so we use Traefik's metrics.
        request_rate = self._query_scalar(
            f'sum(rate(traefik_service_requests_total{{service=~"default-{self.deployment}.*"}}[20s]))'
        )
        p99_latency_sec = self._query_scalar(
            f'histogram_quantile(0.99, sum by (le) '
            f'(rate(traefik_service_request_duration_seconds_bucket{{service=~"default-{self.deployment}.*"}}[20s])))'
        )

        # ── 2. Pod counts and CPU/Memory from Kubernetes API ─────────────────
        total_replicas = 0.0
        ready_replicas = 0.0
        cpu_util = 0.0
        mem_bytes = 0.0

        if self.apps_v1 is not None:
            # 1. Fetch Deployment Replicas
            # 1. Fetch Deployment Replicas using kubectl to bypass Python TLS auth bugs
            try:
                import subprocess
                out = subprocess.check_output(
                    ["kubectl", "get", "deploy", self.deployment, "-n", self.namespace, 
                     "-o", "jsonpath={.status.replicas},{.status.readyReplicas}"],
                    text=True, timeout=5
                ).strip().split(',')
                total_replicas = float(out[0] if out[0] else 0)
                ready_replicas = float(out[1] if len(out) > 1 and out[1] else 0)
            except Exception as e:
                logger.error("Failed to fetch deployment replicas via kubectl: %s", e)

            # 2. Fetch Pod CPU/Memory Metrics
            try:
                cust = client.CustomObjectsApi()
                pod_metrics = cust.list_namespaced_custom_object(
                    "metrics.k8s.io", "v1beta1", self.namespace, "pods"
                )

                for item in pod_metrics.get("items", []):
                    if item["metadata"]["name"].startswith(self.deployment):
                        for container in item["containers"]:
                            # container name in the boutique frontend is "server"
                            if container["name"] in ("server", self.deployment):
                                cpu_str = container["usage"]["cpu"]
                                if cpu_str.endswith("m"):
                                    cpu_util += float(cpu_str[:-1]) / 1000.0
                                elif cpu_str.endswith("n"):
                                    cpu_util += float(cpu_str[:-1]) / 1e9

                                mem_str = container["usage"]["memory"]
                                if mem_str.endswith("Mi"):
                                    mem_bytes += float(mem_str[:-2]) * 1024 * 1024
                                elif mem_str.endswith("Ki"):
                                    mem_bytes += float(mem_str[:-2]) * 1024
                                elif mem_str.endswith("Gi"):
                                    mem_bytes += float(mem_str[:-2]) * 1024 * 1024 * 1024

                # Normalise CPU against per-pod limit (100m for frontend now)
                if total_replicas > 0:
                    cpu_util = cpu_util / (total_replicas * 0.10)
                else:
                    cpu_util = 0.0

            except Exception as e:
                logger.error("Failed to fetch Kubernetes core metrics: %s", e)

        # ── 3. Derived metrics ────────────────────────────────────────────────
        pending_pods = max(0.0, total_replicas - ready_replicas)
        p99_latency_ms = p99_latency_sec * 1000.0 if not np.isnan(p99_latency_sec) else 10.0

        # Queue depth estimate (latency-based, matches simulator approximation)
        base_latency = 10.0
        queue_depth = max(0.0, (p99_latency_ms - base_latency) * request_rate / 1000.0)

        pods_per_node = total_replicas / max(1, self.num_nodes)
        node_cpu_util = (pods_per_node * 0.20) / self.node_cpu   # 200m per pod
        node_mem_util = (pods_per_node * 0.128) / self.node_mem  # 128Mi limit

        cost_rate = max(1.0, math.ceil(total_replicas * 0.20 / (self.node_cpu * 0.85))) * 0.35

        # ── 4. Build & normalise 23-dim vector (same layout as K8sSimEnv) ────
        obs = np.zeros(23, dtype=np.float32)

        obs[0] = min(1.0, max(0.0, cpu_util if not np.isnan(cpu_util) else 0.0))
        obs[1] = min(1.0, max(0.0, (mem_bytes / (1024 ** 3)) / (total_replicas * 0.128) if total_replicas > 0 else 0.0))
        obs[2] = total_replicas / self.max_replicas
        obs[3] = pending_pods / 10.0
        obs[4] = request_rate / 500.0
        obs[5] = (request_rate - self.prev_request_rate) / 100.0
        obs[6] = p99_latency_ms / 1000.0
        obs[7] = 20.0 / 30.0  # per_pod_capacity (fixed: observable in sim, estimated in real)
        obs[8] = queue_depth / 10000.0

        for i in range(self.num_nodes):
            base = 9 + i * 3
            obs[base]     = max(0.0, 1.0 - node_cpu_util)
            obs[base + 1] = max(0.0, 1.0 - node_mem_util)
            obs[base + 2] = pods_per_node / 30.0

        phase = 2.0 * np.pi * self.step_count / 120.0
        obs[18] = np.sin(phase)
        obs[19] = np.cos(phase)
        obs[20] = cost_rate / 2.0
        obs[21] = self.prev_cpu_util

        velocity = request_rate - self.prev_request_rate
        prev_velocity = self.prev_request_rate - self.prev_prev_request_rate
        obs[22] = (velocity - prev_velocity) / 500.0

        # Update stateful history
        self.prev_prev_request_rate = self.prev_request_rate
        self.prev_request_rate = request_rate
        self.prev_cpu_util = obs[0]
        self.step_count += 1

        obs = np.nan_to_num(obs)
        obs = np.clip(obs, -1.0, 10.0)
        return obs

    def _query_scalar(self, query: str) -> float:
        """Execute a PromQL query that returns a single scalar value."""
        try:
            result = self.prom.custom_query(query)
            if result and len(result) > 0 and "value" in result[0]:
                val = result[0]["value"][1]
                if val == "NaN":
                    return float("nan")
                return float(val)
            return 0.0
        except Exception as e:
            logger.error("Prometheus query failed: %s — %s", query, e)
            return 0.0
