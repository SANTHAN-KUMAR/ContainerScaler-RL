# Component: PrometheusObserver

## Responsibility

Queries Prometheus for real cluster metrics and translates them into the same 22-dimensional observation vector the agent was trained on. This is the sim-to-real bridge — it must produce numbers in the same format and scale the simulator used during training, or the agent's policy will behave incorrectly.

---

## Inputs and Outputs

| | Description |
|---|---|
| **Input** | Prometheus HTTP endpoint URL |
| **Input** | Namespace and deployment name to observe |
| **Output** | 22-dim observation vector (numpy array) — identical format to simulator output |

---

## Interface

```
__init__(prom_url, namespace, deployment)  → PrometheusObserver
get_state()                                → ObservationVector (22-dim ndarray)
```

---

## Internal Logic

`get_state()` runs the following in order:

### 1. Query Prometheus
Fire 6 PromQL queries against the Prometheus HTTP API:

| Metric | PromQL Query |
|---|---|
| CPU utilization | `rate(container_cpu_usage_seconds_total{pod=~"podinfo.*"}[1m])` |
| Memory utilization | `container_memory_working_set_bytes{pod=~"podinfo.*"}` |
| Total replicas | `kube_deployment_status_replicas{deployment="podinfo"}` |
| Ready replicas | `kube_deployment_status_ready_replicas{deployment="podinfo"}` |
| Request rate | `rate(http_requests_total{app="podinfo"}[1m])` |
| P99 latency | `histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[1m]))` |

### 2. Derive Pending Pods
```
pending_pods = total_replicas - ready_replicas
```
Prometheus doesn't expose pending pods directly — it's computed from the difference.

### 3. Estimate Queue Depth
The real cluster doesn't expose queue depth directly. It is estimated from latency overshoot:
```
if p99_latency > base_latency:
    estimated_queue = (p99_latency - base_latency) × request_rate / 1000
else:
    estimated_queue = 0
```

### 4. Compute Traffic Derivative
```
derivative = (current_request_rate - previous_request_rate)
```
Stored from the previous call to `get_state()`.

### 5. Normalize All Values
Apply the same normalization the simulator used so values are in the same range the agent was trained on:

```
cpu_util              → already in [0,1]
mem_util              → divide by pod memory limit
replicas              → divide by max_replicas (30)
pending_pods          → divide by 10
request_rate          → divide by 500
derivative            → divide by 100
p99_latency           → divide by 1000
queue_depth           → divide by 10000
cost_rate             → divide by 2.0
```

### 6. Build and Return Vector
Assemble all 22 values in the exact same order as the simulator's observation vector.

---

## Internal State

| Variable | Description |
|---|---|
| `prev_request_rate` | Request rate from previous call (for derivative) |
| `prev_cpu_util` | CPU utilization from previous call (dimension 21) |
| `step_count` | Current step number (for time encoding) |
| `prom_client` | Prometheus HTTP client instance |

---

## Sim-to-Real Gap

This is where training assumptions meet reality. Key differences:

| Simulator Assumption | Reality |
|---|---|
| CPU util is computed instantly | Prometheus uses a 1-minute rate window — values lag by up to 60s |
| Queue depth is tracked exactly | Queue depth is estimated from latency — imprecise |
| Metrics are noise-free | Real metrics have scrape jitter, missing data points, stale values |
| Node metrics are exact | kube-state-metrics has its own scrape interval |

These gaps are documented in the sim-to-real analysis (Experiment 8) but not fully eliminated.

---

## Failure Modes

| Failure | What Happens | Mitigation |
|---|---|---|
| Prometheus unreachable | `get_state()` throws exception | Catch exception → trigger HPA fallback |
| Stale metrics | Prometheus returns old data | Check metric timestamp; if stale > 60s, use last known good values |
| Missing metric | Query returns no data | Use safe default values (e.g., 0 for pending pods) |
| Normalization mismatch | Values outside trained range | Clip all values to [0, 1] before returning |

---

## Dependencies

| Dependency | Purpose |
|---|---|
| `prometheus-api-client` | HTTP client for PromQL queries |
| `numpy` | Vector construction and normalization |
| `kubernetes` | Used to get deployment metadata (replica limits) |
