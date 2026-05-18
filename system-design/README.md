# ContainerScale-RL: System Design

## Table of Contents

- [1. Problem Statement](#1-problem-statement)
- [2. Requirements](#2-requirements)
- [3. System Overview](#3-system-overview)
- [4. Two-Phase Architecture](#4-two-phase-architecture)
- [5. Component Map](#5-component-map)
- [6. Data Flow](#6-data-flow)
- [7. The Sim-to-Real Bridge](#7-the-sim-to-real-bridge)
- [8. Fault Tolerance](#8-fault-tolerance)
- [9. Low-Level Design Index](#9-low-level-design-index)

---

## 1. Problem Statement

Kubernetes' built-in autoscaler (HPA) is a reactive proportional controller. It scales only after overload occurs, ignores pod startup delay, and has no concept of cost. The result is a trilemma: operators must choose two of {reliability, cost efficiency, operational simplicity}.

This system replaces HPA's formula with a Reinforcement Learning agent (RecurrentPPO) that:
- Anticipates traffic trends via LSTM memory
- Accounts for pod cold-start delay as a first-class variable
- Optimizes a multi-objective reward balancing SLA compliance against cost

---

## 2. Requirements

### Functional Requirements

| # | Requirement |
|---|---|
| F1 | Observe cluster state every 30 seconds and produce a replica count decision |
| F2 | Train an RL agent in simulation across 5 traffic patterns |
| F3 | Enforce hard safety constraints on every action (min/max replicas, no scale-down during high latency) |
| F4 | Fall back to HPA if the RL agent fails for any reason |
| F5 | Deploy the trained agent on a live cluster and control it via the Kubernetes API |
| F6 | Run controlled experiments comparing RL vs HPA under identical conditions |
| F7 | Log all metrics during live runs for post-hoc analysis |

### Non-Functional Requirements

| # | Requirement | Target |
|---|---|---|
| NF1 | Simulator throughput | 1000+ steps/second |
| NF2 | Inference latency | < 100ms per decision |
| NF3 | Safety filter coverage | 100% of actions pass through filter |
| NF4 | Fault tolerance | Any RL failure falls back to HPA within one control interval |
| NF5 | Reproducibility | Experiments repeatable with fixed random seeds |
| NF6 | Portability | Runs on local machine, no cloud dependency |
| NF7 | Observability | All decisions and metrics logged for analysis |

---

## 3. System Overview

The system has two completely independent phases joined by a single artifact — the trained model weights file.

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: SIMULATION TRAINING                                    │
│                                                                  │
│  WorkloadGenerator ──► K8sSimEnv ──► RecurrentPPO               │
│       (synthetic          (Gymnasium      (SB3 training          │
│        traffic)            environment)    loop)                 │
│                                │                                 │
│                                ▼                                 │
│                       ppo_autoscaler.zip                         │
│                       (trained weights)                          │
└─────────────────────────────┬───────────────────────────────────┘
                               │
                    model weights file
                    (the only bridge)
                               │
┌─────────────────────────────▼───────────────────────────────────┐
│  PHASE 2: LIVE CLUSTER DEPLOYMENT                                │
│                                                                  │
│  Locust ──► k3s Cluster ──► Prometheus ──► PrometheusObserver   │
│  (traffic)   (podinfo app)   (metrics)      (22-dim vector)      │
│                                                    │             │
│                                                    ▼             │
│                                          ContainerScaleAgent     │
│                                          (RecurrentPPO inference)│
│                                                    │             │
│                                                    ▼             │
│                                            SafetyFilter          │
│                                                    │             │
│                                          ┌─────────┴──────────┐ │
│                                          │                     │ │
│                                    safe delta?            unsafe │
│                                          │                  HPA  │
│                                          ▼                       │
│                                  K8sPatchExecutor                │
│                                  (Kubernetes API)                │
│                                          │                       │
│                                          ▼                       │
│                              replica count changes ──────────────┤
│                              (feeds back into Prometheus)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Two-Phase Architecture

### Phase 1: Simulation Training

The agent trains entirely in a synthetic environment. No real cluster is involved.

**Why simulation first:**
- Training requires millions of steps — impossible on a live cluster without causing outages
- The simulator runs at 1000+ steps/second — a real cluster runs at 1 step per 30 seconds
- Mistakes in simulation are free; mistakes on a live cluster are outages

**What the simulator models:**
- Pod cold-start delay (30–180 seconds per pod)
- Request queue buildup when demand exceeds capacity
- Multi-objective cost (CPU utilization + dollar cost)
- Five distinct traffic patterns

**Output of Phase 1:** A single `.zip` file containing the trained neural network weights.

---

### Phase 2: Live Cluster Deployment

The trained model is loaded and run against a real k3s cluster. The agent makes one decision every 30 seconds — the same interval it was trained on.

**What's real in Phase 2:**
- The cluster (k3s running locally)
- The application (podinfo — a real Go service)
- The traffic (Locust generating HTTP requests)
- The metrics (Prometheus scraping real pod metrics)
- The scaling actions (Kubernetes API patching real deployments)

**What carries over from Phase 1:**
- The trained model weights (frozen — no further learning)
- The 22-dimensional observation format (must match exactly)
- The action space (replica delta -3 to +3)
- The safety filter rules

---

## 5. Component Map

### Phase 1 Components

| Component | File | Role |
|---|---|---|
| WorkloadGenerator | `src/env/workload.py` | Generates synthetic traffic patterns |
| K8sSimEnv | `src/env/k8s_sim.py` | Gymnasium environment — simulates cluster physics |
| Training Script | `src/training/train_ppo.py` | Wires simulator + PPO, runs training loop |

### Phase 2 Components

| Component | File | Role |
|---|---|---|
| LiveClusterAgent | `src/live/live_agent.py` | Top-level control loop coordinator |
| PrometheusObserver | `src/live/observer.py` | Queries Prometheus → builds 22-dim vector |
| ContainerScaleAgent | `src/agents/agent.py` | Loads model, runs inference, manages LSTM state |
| SafetyFilter | `src/safety/safety_filter.py` | Hard-coded invariant rules on every action |
| RealisticHPA | `src/agents/hpa_baseline.py` | Comparison baseline + RL fallback controller |
| K8sPatchExecutor | `src/live/executor.py` | Patches Kubernetes deployment replica count |
| MetricsLogger | `src/live/metrics_logger.py` | Logs all decisions and metrics for analysis |

### Shared Artifact

| Artifact | Description |
|---|---|
| `ppo_autoscaler.zip` | Trained model weights — the bridge between Phase 1 and Phase 2 |

---

## 6. Data Flow

### Phase 1 Data Flow (Training)

```
WorkloadGenerator.get_rate(step)
        │
        │ rps (float)
        ▼
K8sSimEnv.step(action)
        │
        │ Internally computes:
        │   - queue dynamics (arrivals vs departures)
        │   - cold-start aging (pending → ready pods)
        │   - cost calculation (nodes × price)
        │   - reward (SLA + cost + stability + crash)
        │
        │ Returns: (obs[22], reward, done, info)
        ▼
RecurrentPPO
        │
        │ Collects 8 envs × 2048 steps = 16,384 transitions
        │ Runs 10 gradient update epochs
        │ Repeats until 500,000 total steps
        │
        ▼
ppo_autoscaler.zip
```

### Phase 2 Data Flow (Live)

```
Locust → HTTP requests → podinfo (k3s)
                              │
                              │ metrics scraped every 15s
                              ▼
                         Prometheus
                              │
                              │ PromQL queries
                              ▼
                    PrometheusObserver
                              │
                              │ obs[22] (normalized, same format as simulator)
                              ▼
                    ContainerScaleAgent
                              │
                              │ LSTM hidden state maintained across steps
                              │ Neural network inference
                              │
                              │ action integer (0–6)
                              ▼
                        SafetyFilter
                              │
                    ┌─────────┴──────────┐
               safe delta            blocked/modified
                    │                    │
                    │              RealisticHPA.act()
                    │                    │
                    └─────────┬──────────┘
                              │
                              │ target replica count
                              ▼
                    K8sPatchExecutor
                              │
                              │ Kubernetes API patch
                              ▼
                    podinfo deployment
                    (replica count changes)
                              │
                              └──────────────────────► back to Prometheus
                                    (closed feedback loop)
```

---

## 7. The Sim-to-Real Bridge

The 22-dimensional observation vector is the contract between Phase 1 and Phase 2. Every feature the simulator produces must be reproducible from real Prometheus metrics.

| Observation Feature | Simulator Source | Prometheus Source |
|---|---|---|
| cpu_util | Computed from formula | `rate(container_cpu_usage_seconds_total[1m])` |
| mem_util | Computed from formula | `container_memory_working_set_bytes` |
| replicas | Internal counter | `kube_deployment_status_replicas` |
| pending_pods | Pending list length | `total_replicas - ready_replicas` |
| request_rate | WorkloadGenerator | `rate(http_requests_total[1m])` |
| p99_latency | Queue model formula | `histogram_quantile(0.99, ...)` |
| queue_depth | Exact internal value | Estimated from latency overshoot |

**Known gaps:**
- Prometheus metrics lag by up to 60 seconds (rate windows)
- Queue depth is estimated in live deployment, exact in simulation
- Real pod startup times differ from simulator's LogNormal model

These gaps are documented and measured in Experiment 8 (sim-to-real gap analysis).

---

## 8. Fault Tolerance

The system is designed to never stop making scaling decisions, even under failure.

| Failure Scenario | Response |
|---|---|
| RL model file missing | Fail fast at startup with clear error |
| Prometheus unreachable | Skip step, retry next interval |
| RL inference exception | Fall back to HPA for that step |
| Kubernetes API down | Log error, keep current replica count |
| Safety filter blocks all actions | Cluster stays at current replicas (safe default) |

The fallback chain is: **RL agent → HPA formula → no change (hold current replicas)**

No failure mode results in the cluster being left with 0 replicas or an uncontrolled state.

---

## 9. Low-Level Design Index

Detailed component design documents:

| Component | Document |
|---|---|
| K8sSimEnv (Simulator) | [components/simulator.md](components/simulator.md) |
| WorkloadGenerator | [components/workload-generator.md](components/workload-generator.md) |
| Training Script | [components/training-script.md](components/training-script.md) |
| PrometheusObserver | [components/prometheus-observer.md](components/prometheus-observer.md) |
| ContainerScaleAgent | [components/container-scale-agent.md](components/container-scale-agent.md) |
| SafetyFilter | [components/safety-filter.md](components/safety-filter.md) |
| RealisticHPA | [components/hpa-baseline.md](components/hpa-baseline.md) |
| K8sPatchExecutor | [components/kubernetes-executor.md](components/kubernetes-executor.md) |
| LiveClusterAgent | [components/live-cluster-agent.md](components/live-cluster-agent.md) |
