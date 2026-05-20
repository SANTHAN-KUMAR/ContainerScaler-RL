"""
K8sPatchExecutor — Interacts with Kubernetes API.

Translates the agent's scaling decision into a real Kubernetes API call
that changes the number of running pods via Deployment patching.
"""

from __future__ import annotations

import logging

from kubernetes import client, config

logger = logging.getLogger(__name__)


class K8sPatchExecutor:
    """Executes scaling actions on a live Kubernetes cluster.

    Parameters
    ----------
    namespace : str
        Kubernetes namespace.
    deployment : str
        Target deployment name.
    min_replicas : int
        Hard minimum replica count.
    max_replicas : int
        Hard maximum replica count.
    """

    def __init__(
        self,
        namespace: str = "default",
        deployment: str = "podinfo",
        min_replicas: int = 2,
        max_replicas: int = 30,
    ) -> None:
        self.namespace = namespace
        self.deployment = deployment
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas

        # Load kube config (works in-cluster or locally via ~/.kube/config)
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config.")
        except config.ConfigException:
            try:
                config.load_kube_config()
                logger.info("Loaded local kubeconfig.")
            except Exception as e:
                logger.error(f"Failed to load kubeconfig: {e}")
                raise

        self.apps_v1 = client.AppsV1Api()

    def scale(self, target_replicas: int | float) -> None:
        """Patch the deployment with the target replica count.

        Parameters
        ----------
        target_replicas : int | float
            Desired absolute number of replicas.
        """
        # 1. Clamp target
        target = max(self.min_replicas, min(self.max_replicas, int(target_replicas)))

        # 2. Build patch body
        patch_body = {"spec": {"replicas": target}}

        # 3. Call Kubernetes API
        try:
            logger.info(
                f"Patching Deployment {self.namespace}/{self.deployment} "
                f"to {target} replicas."
            )
            self.apps_v1.patch_namespaced_deployment_scale(
                name=self.deployment,
                namespace=self.namespace,
                body=patch_body,
            )
        except Exception as e:
            logger.error(f"Failed to patch deployment {self.deployment}: {e}")
            # We log and swallow the exception so the control loop continues.
            # Next iteration might succeed.
