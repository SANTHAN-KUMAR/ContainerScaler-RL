# Component: LiveClusterAgent

## Responsibility

Orchestrates the Phase 2 control loop. Every 30 seconds, it collects cluster state from Prometheus, runs it through the RL agent, applies the safety filter, and patches the Kubernetes deployment. It is the top-level coordinator for live deployment — it owns the control loop and wires all Phase 2 components together.

---

## Inputs and Outputs

| | Description |
|---|---|
| **Input** | Prometheus URL, namespace, deployment name |
| **Input** | Duration (number of steps to run) |
| **Output** | Scaling actions applied to the cluster (side effects) |
| **Output** | Metrics log (for post-hoc analysis) |

---

## Interface

```
__init__(prom_url, namespace, deployment)    → LiveClusterAgent
run_loop(duration_steps=120)                 → None
```

---

## The Control Loop

`run_loop()` runs this sequence every 30 seconds for `duration_steps` iterations:

```
for step in 0 to duration_steps:

    1. state = observer.get_state()
       → Query Prometheus, build 22-dim vector

    2. action, lstm_states = ppo.predict(state, lstm_states)
       → Run neural network inference

    3. delta = action - 3
       → Map [0..6] to [-3..+3]

    4. safe_delta = safety.check(state, delta, step)
       → Apply hard safety rules

    5. if safe_delta != 0:
           target = state.replicas + safe_delta
           executor.scale(target)
           → Patch Kubernetes deployment

    6. logger.log(step, state, delta, safe_delta)
       → Record everything for analysis

    7. sleep(30 seconds)
       → Wait for next control interval
```

---

## Feedback Loop

The control loop is closed — the executor changes the cluster, Prometheus observes those changes, and the observer picks them up on the next iteration. This means:

- A scale-up at step N will show up as increased `ready_replicas` at step N+2 or N+3 (after pod startup)
- The agent's LSTM hidden state carries the memory of having scaled up, so it doesn't immediately scale up again

This closed-loop behavior is what makes the live deployment fundamentally different from the simulator — the agent's actions have real consequences that feed back into its next observation.

---

## Timing

| Event | Timing |
|---|---|
| Control loop interval | 30 seconds (matches simulator step size) |
| Prometheus scrape interval | 15 seconds (configured in Prometheus) |
| Pod startup time | 30–180 seconds (real k3s behavior) |
| Full experiment duration | 120 steps × 30s = 1 hour |

The 30-second control interval matches the simulator's step size exactly. This is intentional — the agent was trained assuming 30 seconds between decisions, so the live loop must match.

---

## Metrics Logging

Every step is logged for post-hoc analysis and comparison against HPA:

| Metric | Description |
|---|---|
| `timestamp` | Wall clock time |
| `step` | Step number within the run |
| `p99_latency` | Observed tail latency (ms) |
| `request_rate` | Observed traffic (rps) |
| `replicas` | Current replica count |
| `pending_pods` | Pods still starting |
| `cpu_util` | CPU utilization |
| `proposed_delta` | What the RL agent wanted to do |
| `safe_delta` | What the safety filter allowed |
| `cost_rate` | Estimated $/hour |
| `sla_breach` | Boolean — was P99 > SLA target this step? |

These logs are what Experiment 8 uses to compare RL vs HPA.

---

## Internal State

| Variable | Description |
|---|---|
| `ppo` | Loaded RecurrentPPO model |
| `lstm_states` | Current LSTM hidden state |
| `observer` | PrometheusObserver instance |
| `executor` | K8sPatchExecutor instance |
| `safety` | SafetyFilter instance |
| `logger` | MetricsLogger instance |

---

## Failure Modes

| Failure | What Happens | Mitigation |
|---|---|---|
| Prometheus down | `get_state()` throws | Log error, skip this step, retry next interval |
| RL inference fails | Exception in `ppo.predict()` | Fall back to HPA for this step |
| Kubernetes API down | `executor.scale()` throws | Log error, continue loop — cluster keeps current replicas |
| Loop runs too slow | Step takes > 30s | Log warning; next sleep is shortened to compensate |

---

## Dependencies

| Dependency | Purpose |
|---|---|
| `PrometheusObserver` | Cluster state collection |
| `ContainerScaleAgent` | RL inference + HPA fallback |
| `SafetyFilter` | Action validation |
| `K8sPatchExecutor` | Kubernetes API patching |
| `MetricsLogger` | Experiment data collection |
| `time` | 30-second sleep between steps |
