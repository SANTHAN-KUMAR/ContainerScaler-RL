"""
PrometheusObserver — The Sim-to-Real bridge.

Queries real cluster metrics and normalizes them into the exact same
22-dimensional observation vector the agent saw during simulation training.
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
        Target deployment name (e.g., "podinfo").
    """

    def __init__(
        self,
        prom_url: str = "http://localhost:9090",
        namespace: str = "default",
        deployment: str = "podinfo",
    ) -> None:
        self.prom = PrometheusConnect(url=prom_url, disable_ssl=True)
        self.namespace = namespace
        self.deployment = deployment

        # Needed for max_replicas and pod limits if not hard-coded
        try:
            config.load_kube_config()
            self.apps_v1 = client.AppsV1Api()
        except Exception:
            logger.warning("Could not load kubeconfig, using defaults for cluster capacity.")
            self.apps_v1 = None

        # Fixed limits (should match simulator config)
        self.max_replicas = 30
        self.node_cpu = 4.0
        self.node_mem = 16.0
        self.num_nodes = 3

        # Stateful metrics
        self.prev_request_rate: float = 0.0
        self.prev_cpu_util: float = 0.0
        self.step_count: int = 0

    def get_state(self) -> np.ndarray:
        """Query Prometheus and build the 22-dim normalized observation vector.

        Returns
        -------
        np.ndarray
            Observation vector matching simulator format.
        """
        # 1. Prometheus Queries for HTTP Traffic ONLY
        request_rate = self._query_scalar(
            f'sum(rate(http_requests_total{{app="{self.deployment}"}}[1m]))'
        )
        # Assuming histogram bucket format for p99 latency
        p99_latency_sec = self._query_scalar(
            f'histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket{{app="{self.deployment}"}}[1m])))'
        )

        # 2. Native Kubernetes API for Structural Metrics (CPU, Mem, Replicas)
        total_replicas = 0.0
        ready_replicas = 0.0
        cpu_util = 0.0
        mem_bytes = 0.0

        if self.apps_v1:
            try:
                # Fetch Replicas
                deployment_info = self.apps_v1.read_namespaced_deployment(self.deployment, self.namespace)
                total_replicas = float(deployment_info.status.replicas or 0)
                ready_replicas = float(deployment_info.status.ready_replicas or 0)
                
                # Fetch CPU/Memory from metrics.k8s.io (cAdvisor equivalent)
                cust = client.CustomObjectsApi()
                pod_metrics = cust.list_namespaced_custom_object("metrics.k8s.io", "v1beta1", self.namespace, "pods")
                
                for item in pod_metrics.get('items', []):
                    if item['metadata']['name'].startswith(self.deployment):
                        for container in item['containers']:
                            if container['name'] == 'podinfod':
                                # Parse CPU (e.g. '15m' or '15000n')
                                cpu_str = container['usage']['cpu']
                                if cpu_str.endswith('m'):
                                    cpu_util += float(cpu_str[:-1]) / 1000.0
                                elif cpu_str.endswith('n'):
                                    cpu_util += float(cpu_str[:-1]) / 1e9
                                
                                # Parse Memory (e.g. '150Mi' or '150000Ki')
                                mem_str = container['usage']['memory']
                                if mem_str.endswith('Mi'):
                                    mem_bytes += float(mem_str[:-2]) * 1024 * 1024
                                elif mem_str.endswith('Ki'):
                                    mem_bytes += float(mem_str[:-2]) * 1024
                
                # Normalize CPU against the expected total limits for the deployment (250m per pod)
                if total_replicas > 0:
                    cpu_util = cpu_util / (total_replicas * 0.25)
                else:
                    cpu_util = 0.0

            except Exception as e:
                logger.error(f"Failed to fetch Kubernetes core metrics: {e}")

        # 2. Derived metrics
        pending_pods = max(0.0, total_replicas - ready_replicas)
        p99_latency_ms = p99_latency_sec * 1000.0 if not np.isnan(p99_latency_sec) else 10.0

        # Estimate queue depth (real cluster doesn't expose this directly)
        base_latency = 10.0
        queue_depth = max(0.0, (p99_latency_ms - base_latency) * request_rate / 1000.0)

        derivative = request_rate - self.prev_request_rate

        # Nodes logic (simplified for single local k3s node in experiment,
        # but structured to match simulator's 3-node layout)
        pods_per_node = total_replicas / max(1, self.num_nodes)
        node_cpu_util = (pods_per_node * 0.25) / self.node_cpu
        node_mem_util = (pods_per_node * 0.5) / self.node_mem
        
        cost_rate = max(1.0, math.ceil(total_replicas * 0.25 / (self.node_cpu * 0.85))) * 0.35 # Approx node price

        # 3. Build & Normalize vector
        obs = np.zeros(22, dtype=np.float32)
        
        obs[0] = min(1.0, max(0.0, cpu_util if not np.isnan(cpu_util) else 0.0))
        # Appx mem util (512Mi limit)
        obs[1] = min(1.0, max(0.0, (mem_bytes / 1024/1024/1024) / (total_replicas * 0.5) if total_replicas > 0 else 0.0))
        obs[2] = total_replicas / self.max_replicas
        obs[3] = pending_pods / 10.0
        obs[4] = request_rate / 500.0
        obs[5] = derivative / 100.0
        obs[6] = p99_latency_ms / 1000.0
        obs[7] = 20.0 / 30.0  # Estimated per_pod_capacity (observable in sim, fixed in real)
        obs[8] = queue_depth / 10000.0

        for i in range(self.num_nodes):
            base = 9 + i * 3
            obs[base] = max(0.0, 1.0 - node_cpu_util)
            obs[base + 1] = max(0.0, 1.0 - node_mem_util)
            obs[base + 2] = pods_per_node / 30.0

        phase = 2.0 * np.pi * self.step_count / 120.0
        obs[18] = np.sin(phase)
        obs[19] = np.cos(phase)
        obs[20] = cost_rate / 2.0
        obs[21] = self.prev_cpu_util

        # Update state
        self.prev_request_rate = request_rate
        self.prev_cpu_util = obs[0]
        self.step_count += 1

        # Clip all values to [0, 1] range except those that can be negative
        obs = np.nan_to_num(obs)
        obs = np.clip(obs, -1.0, 10.0)

        return obs

    def _query_scalar(self, query: str) -> float:
        """Execute a PromQL query that returns a single scalar value."""
        try:
            result = self.prom.custom_query(query)
            if result and len(result) > 0 and 'value' in result[0]:
                val = result[0]['value'][1]
                if val == 'NaN':
                    return float('nan')
                return float(val)
            return 0.0
        except Exception as e:
            logger.error(f"Prometheus query failed: {query} - {e}")
            return 0.0
