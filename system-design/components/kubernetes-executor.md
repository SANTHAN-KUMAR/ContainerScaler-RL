# Component: K8sPatchExecutor

## Responsibility

Translates the agent's scaling decision (a replica delta integer) into a real Kubernetes API call that changes the number of running pods. The only component that directly mutates cluster state.

---

## Inputs and Outputs

| | Description |
|---|---|
| **Input** | Target replica count (absolute number, not delta) |
| **Input** | Namespace and deployment name |
| **Output** | None (side effect: Kubernetes deployment is patched) |

---

## Interface

```
__init__(namespace, deployment)    → K8sPatchExecutor
scale(target_replicas)             → None
```

---

## Internal Logic

### `scale(target_replicas)`

```
1. Clamp target: target = max(2, min(30, int(target_replicas)))
2. Build patch body: {"spec": {"replicas": target}}
3. Call Kubernetes API:
   apps_v1.patch_namespaced_deployment_scale(
       deployment, namespace, body
   )
4. Log the scaling action with timestamp
```

Note: the executor receives an **absolute replica count**, not a delta. The conversion from delta to absolute happens in the LiveClusterAgent before calling `scale()`:

```
target = state.current_replicas + safe_delta
executor.scale(target)
```

---

## What Happens After the API Call

The executor's job ends at the API call. Kubernetes takes over:

1. Kubernetes scheduler assigns new pods to nodes
2. Kubelet on each node pulls the container image and starts the pod
3. Pod goes through: Pending → ContainerCreating → Running → Ready
4. Once Ready, the pod starts receiving traffic
5. Prometheus picks up the new replica count on next scrape

The executor does not wait for pods to become ready — it fires and forgets. The agent will observe the new replica count (including pending pods) on the next `get_state()` call.

---

## Why Absolute Count, Not Delta

The Kubernetes API accepts an absolute replica count, not a delta. Passing absolute count also makes the executor idempotent — calling `scale(5)` twice has the same effect as calling it once. This is safer than delta-based patching, which could compound errors if called multiple times.

---

## Internal State

| Variable | Description |
|---|---|
| `apps_v1` | Kubernetes AppsV1Api client instance |
| `namespace` | Target namespace (e.g., "default") |
| `deployment` | Target deployment name (e.g., "podinfo") |

---

## Failure Modes

| Failure | What Happens | Mitigation |
|---|---|---|
| API server unreachable | Exception thrown | Caught by LiveClusterAgent → HPA fallback |
| Insufficient node capacity | Pods stay Pending indefinitely | Node autoscaler handles this (out of scope) |
| Permission denied | 403 from API server | Check RBAC — service account needs patch permission on deployments |
| Invalid replica count | API rejects request | Pre-clamp to [2, 30] before calling API |

---

## Required Kubernetes RBAC

The service account running the agent needs this permission:

```yaml
rules:
- apiGroups: ["apps"]
  resources: ["deployments/scale"]
  verbs: ["get", "patch"]
```

Without this, every `scale()` call will fail with a 403 error.

---

## Dependencies

| Dependency | Purpose |
|---|---|
| `kubernetes` (Python client) | Kubernetes API access |
| `kubernetes.config` | Loads kubeconfig or in-cluster config |
