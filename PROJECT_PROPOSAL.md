# ContainerScale-RL: Reinforcement Learning for Kubernetes Autoscaling

> An RL agent that learns to autoscale Kubernetes deployments, trained in simulation and deployed on a live cluster. Directly compared against the standard Horizontal Pod Autoscaler.

---

## 1. Problem

Kubernetes HPA is a reactive proportional controller:

```
desiredReplicas = ceil(currentReplicas × currentCPU / targetCPU)
```

It has four structural defects:

| Defect | What Happens | Real Cost |
|---|---|---|
| **Reactive** | Scales only after overload. No trend, no memory, no anticipation. | SLA breaches during every traffic ramp. |
| **Cold-start blind** | Assumes new pods serve instantly. Real startup: 30–180 seconds. | Queue buildup and latency spikes during the startup gap. |
| **Cost-blind** | Optimizes CPU utilization only. No concept of money. | Operators over-provision to 20–40% utilization as insurance. 60–80% of capacity sits idle. |
| **Placement-decoupled** | Decides *how many* pods. A separate scheduler decides *where*. Neither coordinates with the other. | Out of scope for this project. |

The industry compensates with structural over-provisioning, manual pre-scaling runbooks, cron-based workarounds, and SRE intervention. The result is a trilemma: pick two of {reliability, cost efficiency, operational simplicity}. None of these are solutions. They are compensations for a controller that cannot represent the problem it governs.

### What This Project Does

Addresses defects 1–3 with a model-free RL agent (RecurrentPPO) that:
- Learns temporal traffic patterns via LSTM memory (anticipation)
- Trains on a simulator that models cold-start delay as a first-class state variable (cold-start awareness)
- Optimizes a multi-objective reward balancing SLA compliance against cost (cost awareness)

The agent is trained in simulation, then deployed on a real local Kubernetes cluster and directly compared against standard HPA under identical workloads.

### Honest Scope

| Defect | Coverage | Mechanism |
|---|---|---|
| Cost-blind | **Fully addressed** | Cost term in reward; tunable tradeoff |
| Cold-start blind | **Substantially addressed** | Simulator models pending→ready; LSTM learns pre-scaling |
| Reactive | **Partially addressed** | LSTM learns temporal patterns; not a general forecaster |
| Placement | **Not addressed** | Out of scope |

---

## 2. Architecture Overview

The system has two phases: train in simulation, deploy on a live cluster.

```
PHASE 1: SIMULATION TRAINING              PHASE 2: LIVE CLUSTER
┌────────────────────────────┐            ┌─────────────────────────────┐
│  Gymnasium Environment      │            │  Local k3s Cluster           │
│  ┌────────────┐             │            │                             │
│  │  k8s_sim   │──► RecurrentPPO         │  podinfo ← Locust (load)   │
│  │ (queue,    │◄── .learn()  │          │  Prometheus (metrics)       │
│  │  cold-start,│             │            │                             │
│  │  cost,     │             │            │  ┌───────────────────────┐  │
│  │  5 traffic │             │  ─model──► │  │   LiveClusterAgent    │  │
│  │  patterns) │             │            │  │   Prom → PPO → K8s   │  │
│  └────────────┘             │            │  └───────────────────────┘  │
│                              │            │                             │
│  7 Simulation Experiments   │            │  Experiment 8: HPA vs RL   │
└────────────────────────────┘            └─────────────────────────────┘
```

**At inference, the agent is simple:** observe metrics → RecurrentPPO decides → safety filter validates → patch replica count. On any failure, fall back to HPA.

---

## 3. Component 1: Simulator (Gymnasium Environment)

The simulator is where the agent trains. It must be fast (1000+ steps/sec), directionally correct, and model the three defects HPA ignores.

### 3.1 MDP Formulation

| Element | Specification |
|---|---|
| **State** | 22-dimensional vector (see below) |
| **Action** | Discrete(7): replica delta ∈ {-3, -2, -1, 0, +1, +2, +3} |
| **Step** | 30 simulated seconds |
| **Episode** | 120 steps = 1 hour |
| **Transition** | Deterministic physics + stochastic noise |

### 3.2 State Vector (22 dimensions)

```python
# Per-deployment (9 dims)
cpu_util                            # current CPU utilization [0,1]
mem_util                            # current memory utilization [0,1]
replicas / max_replicas             # normalized replica count
pending_pods / 10                   # pods still starting up
request_rate / 500                  # current traffic (rps)
(request_rate - prev_rate) / 100    # traffic derivative (trend signal)
p99_latency / 1000                  # tail latency (ms, normalized)
per_pod_capacity / 30               # rps each pod can handle (observable)
queue_depth / 10000                 # accumulated request backlog

# Per-node × 3 nodes (9 dims)
cpu_available / cpu_total           # per-node CPU headroom
mem_available / mem_total           # per-node memory headroom
pod_count / 30                      # per-node pod density

# Global (4 dims)
sin(2π · t/120), cos(2π · t/120)   # time-of-day encoding
cost_rate / 2.0                     # current $/hour
prev_cpu_util                       # previous step CPU (for trend)
```

**Why these features matter:**
- `pending_pods` — the agent can see cold-start in progress and learn to pre-scale
- `queue_depth` — accumulated backlog, not just instantaneous utilization
- `request_rate` derivative — trend signal for anticipation
- `per_pod_capacity` — randomized per episode; agent must adapt to different pod capabilities
- Time encoding — enables learning diurnal patterns

### 3.3 Transition Dynamics

Each step simulates 30 seconds of cluster operation:

**Scaling logic:**
- Scale-up: new pods enter a pending list with a startup time sampled from `LogNormal(mean=cold_start_mean, σ=0.5)`, clipped to [30s, 180s]
- Scale-down: cancel pending pods first (cheaper), then remove ready pods
- Pending pods age 30s per step; when remaining time ≤ 0, they graduate to ready

**Queue dynamics (Little's Law):**
```
service_rate = ready_pods × per_pod_capacity
arrivals_this_step = request_rate × 30
departures_this_step = service_rate × 30
queue_depth = max(0, queue_depth + arrivals - departures)
queue_wait = (queue_depth / request_rate) × 1000   # ms
p99_latency = base_latency + queue_wait + noise
```

This models request accumulation — when demand exceeds capacity, the queue grows and latency degrades continuously, not just instantaneously.

**Cost model (node-level):**
```
cpu_per_pod = 0.25 (250m)
nodes_needed = ceil(total_pod_cpu / (node_cpu × 0.85))
cost_rate = nodes_needed × node_price_per_hour
```

### 3.4 Reward Function

Multi-objective, context-conditioned:

```python
def compute_reward(self, delta):
    # SLA penalty: proportional to latency overshoot
    sla_violation = max(0, self.p99_latency - self.sla_target) / self.sla_target
    r_sla = -10.0 * sla_violation

    # Cost penalty
    r_cost = -0.1 * self.cost_rate

    # Crash penalty: catastrophic latency
    r_crash = -50.0 if self.p99_latency > 5 * self.sla_target else 0

    # Stability: penalize churn ONLY when latency is comfortable
    # (don't punish emergency scaling during overload)
    if self.p99_latency < 0.5 * self.sla_target:
        r_stability = -0.3 * abs(delta)
    else:
        r_stability = 0.0

    return r_sla + r_cost + r_stability + r_crash
```

The context-conditioning is critical: without it, the agent is penalized for scaling up during a spike, creating a perverse incentive to under-provision.

### 3.5 Domain Randomization

Each episode randomizes cluster parameters so the agent generalizes:

```python
def reset(self):
    self.per_pod_capacity = np.random.uniform(15, 30)   # rps/pod
    self.cold_start_mean = np.random.uniform(40, 100)    # seconds
    self.node_price = np.random.uniform(0.20, 0.50)      # $/hour
    self.workload = WorkloadGenerator()                   # random pattern
```

### 3.6 Workload Patterns

Five traffic shapes, randomly selected per episode:

| Pattern | Shape | Peak RPS |
|---|---|---|
| Steady | Flat ~100 rps + Gaussian noise | 100 |
| Diurnal | Sine wave, 50→200→50 | 200 |
| Flash crowd | Baseline 100, spike to 450 at t=0.3 | 450 |
| Gradual ramp | Linear 50→450 over episode | 450 |
| Noisy | Log-normal (mean=100, σ=0.7) | Variable |

Flash crowd peak is 450 rps (not higher) because at the minimum randomized `per_pod_capacity=15`, the agent needs `ceil(450/15)=30` replicas — exactly at `max_replicas`. This ensures every workload is physically serviceable within the action bounds.

---

## 4. Component 2: RL Agent (RecurrentPPO)

### 4.1 Why RecurrentPPO

The autoscaling problem is partially observable: the agent sees metrics with delay, doesn't observe all cluster internals, and must infer temporal patterns. An LSTM policy maintains hidden state across steps, learning to recognize patterns like "traffic is rising" or "I just scaled up and pods are still starting."

A feedforward MLP policy sees each step in isolation and cannot anticipate. Experiment 2 directly tests this.

### 4.2 Training Configuration

```python
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv

model = RecurrentPPO(
    "MlpLstmPolicy",
    SubprocVecEnv([make_env(i) for i in range(8)]),  # 8 parallel envs
    learning_rate=3e-4,
    n_steps=2048,         # steps per rollout per env
    batch_size=64,
    n_epochs=10,          # PPO update epochs per rollout
    gamma=0.99,           # discount factor
    policy_kwargs=dict(
        net_arch=[256, 256],      # 2 hidden layers
        lstm_hidden_size=128,     # LSTM memory capacity
    ),
    tensorboard_log="./logs/ppo/"
)
model.learn(total_timesteps=500_000)
model.save("ppo_autoscaler")
```

**8 parallel environments** ensure diverse workload coverage per training batch. **Standard SB3 on-policy training** — no custom training loops, no replay buffers, no Dyna. The simulator is fast enough that sample efficiency from model-based methods is unnecessary.

### 4.3 Inference Agent

```python
class ContainerScaleAgent:
    def __init__(self):
        self.ppo = RecurrentPPO.load("ppo_autoscaler")
        self.safety = SafetyFilter()
        self.hpa = RealisticHPA()        # fallback
        self.lstm_states = None

    def decide(self, state, sim_step):
        try:
            action, self.lstm_states = self.ppo.predict(
                state.as_vector(),
                state=self.lstm_states,      # maintain LSTM memory
                deterministic=True
            )
            delta = action - 3               # map [0..6] → [-3..+3]
            return self.safety.check(state, delta, sim_step)
        except Exception:
            return self.hpa.act(state)       # any failure → HPA

    def reset(self):
        self.lstm_states = None
        self.safety = SafetyFilter()
        self.hpa.reset()
```

---

## 5. Component 3: Safety Filter

Hard-coded invariant rules. Never trained. Never overridden by the agent.

```python
class SafetyFilter:
    MIN_REPLICAS = 2
    MAX_REPLICAS = 30
    MAX_DELTA = 3
    MIN_INTERVAL_STEPS = 1    # 1 step = 30s cooldown

    def check(self, state, proposed_delta, sim_step):
        # Rule 1: Absolute replica bounds
        new = state.replicas + proposed_delta
        if new < 2 or new > 30:
            proposed_delta = clip(new, 2, 30) - state.replicas

        # Rule 2: No scale-down during high latency
        if proposed_delta < 0 and state.p99_latency > 0.8 * state.sla_target:
            proposed_delta = 0

        # Rule 3: Max delta per step
        proposed_delta = clip(proposed_delta, -3, +3)

        # Rule 4: Rate limiting (sim time, not wall clock)
        if sim_step - self.last_action_step < 1 and proposed_delta != 0:
            proposed_delta = 0

        return proposed_delta
```

**Why this matters:** Early in training, PPO proposes wild actions. The safety filter prevents catastrophic cluster states. As training converges, the agent learns to propose safe actions and the filter becomes redundant. Experiment 4 measures this via regret analysis.

---

## 6. Component 4: HPA Baseline

A realistic HPA implementation matching actual Kubernetes behavior:

```python
class RealisticHPA:
    """HPA with 3-minute scale-down stabilization window."""
    def __init__(self, target_cpu=0.50):
        self.target_cpu = target_cpu
        self.replica_history = deque(maxlen=6)  # 6 steps × 30s = 3 min

    def act(self, state):
        desired = ceil(state.replicas * state.cpu_util / self.target_cpu)
        desired = clip(desired, 2, 30)
        self.replica_history.append(desired)
        # Scale-down uses max of recent window (real K8s behavior)
        if desired < state.replicas:
            desired = max(self.replica_history)
        return int(desired - state.replicas)
```

The stabilization window prevents the comparison from being against a strawman. Real HPA delays scale-down; our baseline does too.

---

## 7. Component 5: World Model (Experiment Only)

A structured world model is trained and evaluated in **Experiment 3 only**. It is not part of the agent's decision path.

### Purpose

Test whether encoding domain knowledge (replicas → capacity → utilization → latency) into model structure improves prediction generalization vs a flat MLP.

### Architecture

5 sub-networks, each predicting one variable from its causal parents:

```
predict_cpu(current_cpu, request_rate, new_replicas) → next_cpu
predict_mem(current_mem, request_rate, new_replicas) → next_mem
predict_pending(current_pending, delta)              → next_pending
predict_request_rate(current_rate, derivative, time)  → next_rate
predict_latency(pred_cpu, pred_mem, pred_pending, queue_depth) → next_latency
```

Each sub-network is a `GaussianMLP` outputting mean + log-variance (probabilistic). An ensemble of 5 models provides epistemic uncertainty via inter-model variance.

### Why Not Use It in the Agent?

The simulator already IS a world model — it runs at 1000+ steps/sec and is fully controllable. A learned approximation of a simulator you already have adds complexity without value. Prior attempts to integrate the world model into the agent (via inference-time override or Dyna-Q training) introduced critical architectural bugs. Keeping it as an experiment avoids these issues while still generating an interesting research contribution.

---

## 8. Phase 2: Live Cluster Deployment

### 8.1 Infrastructure (Zero Cost)

| Component | Tool | Purpose |
|---|---|---|
| Cluster | k3s (local, single-node) | Full K8s API, zero cloud cost |
| Target app | podinfo | Lightweight Go app with /metrics endpoint |
| Load generator | Locust | Scriptable traffic patterns matching simulator |
| Metrics | Prometheus + kube-state-metrics | CPU, memory, replicas, latency, request rate |
| RL agent | Python process (external) | Queries Prometheus, patches Deployment |

### 8.2 Live Agent

```python
class LiveClusterAgent:
    def __init__(self, prom_url, namespace, deployment):
        self.ppo = RecurrentPPO.load("ppo_autoscaler")
        self.safety = SafetyFilter()
        self.observer = PrometheusObserver(prom_url)
        self.executor = K8sPatchExecutor(namespace, deployment)
        self.lstm_states = None

    def run_loop(self, duration_steps=120):
        for step in range(duration_steps):
            state = self.observer.get_state()  # Prometheus → obs vector
            action, self.lstm_states = self.ppo.predict(
                state.as_vector(), state=self.lstm_states, deterministic=True
            )
            delta = action - 3
            safe_delta = self.safety.check(state, delta, step)
            if safe_delta != 0:
                self.executor.scale(state.replicas + safe_delta)
            time.sleep(30)
```

### 8.3 Prometheus Observer

Maps real cluster metrics to the same 22-dim observation vector the simulator uses:

| Simulator Feature | Prometheus Query |
|---|---|
| cpu_util | `rate(container_cpu_usage_seconds_total{pod=~"podinfo.*"}[1m])` |
| mem_util | `container_memory_working_set_bytes{pod=~"podinfo.*"}` |
| replicas | `kube_deployment_status_replicas{deployment="podinfo"}` |
| ready_pods | `kube_deployment_status_ready_replicas{deployment="podinfo"}` |
| request_rate | `rate(http_requests_total{app="podinfo"}[1m])` |
| p99_latency | `histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[1m]))` |

### 8.4 K8s Executor

Patches the Deployment replica count via the Kubernetes API:

```python
class K8sPatchExecutor:
    def scale(self, target_replicas):
        target = max(2, min(30, int(target_replicas)))
        self.apps_v1.patch_namespaced_deployment_scale(
            self.deployment, self.namespace,
            body={"spec": {"replicas": target}}
        )
```

### 8.5 Locust Workload Profiles

Locust scripts mirror the 5 simulator patterns so the trained policy sees familiar traffic shapes on the live cluster. Each profile modulates concurrent user count over time to produce the target RPS curve.

---

## 9. Experiments

### Simulation Experiments (1–7)

| # | Question | Method | Key Metric | Expected |
|---|---|---|---|---|
| 1 | Does RL beat HPA? | RecurrentPPO vs RealisticHPA across all 5 workloads | Cost, SLA violation %, P99 | RL wins on cost and flash-crowd SLA |
| 2 | Does LSTM memory help? | RecurrentPPO vs MlpPolicy PPO | Flash crowd SLA, generalization | LSTM anticipates ramps; MLP can't |
| 3 | Does structured world model generalize? | Structured vs flat MLP prediction on held-out workloads | NLL, calibration error | Structured generalizes better |
| 4 | Does the agent learn safety? | Regret analysis: reward(proposed) - reward(filtered) | Mean regret | Near-zero at convergence |
| 5 | How tunable is the cost-SLA tradeoff? | Vary α_cost ∈ {0.01, 0.1, 1.0} | Pareto frontier | Smooth tradeoff curve |
| 6 | Is uncertainty calibrated? | Ensemble predicted variance vs actual prediction error | Scatter plot correlation | Positive correlation |
| 7 | Does domain randomization help generalize? | Train on {steady, diurnal}, test on {flash, ramp} | SLA on unseen patterns | Randomization helps |

### Live Cluster Experiment (8)

**Protocol:** For each workload pattern, run two 1-hour sessions on the same k3s cluster with the same podinfo deployment:

- **Run A (HPA):** Standard HPA (targetCPU=50%), Locust generates traffic
- **Run B (RL):** RL agent controls replicas (no HPA), identical Locust traffic

**Metrics compared:**

| Metric | Source |
|---|---|
| SLA breach % | % of 30s windows where P99 > target |
| Cost proxy | active_nodes × price/hour |
| Cold-start recovery time | Time from scale-up to P99 < target |
| Scaling events / hour | Count of non-zero deltas |
| Replica trajectory | Time series of replica count |

**Expected:** RL outperforms on diurnal/flash/ramp (fewer SLA breaches, faster recovery). Ties or marginal improvement on steady state. A sim-to-real gap analysis documents where the policy's simulator assumptions break.

### Experiment 4 Detail: Regret-Based Safety Measurement

For each step where the safety filter modifies PPO's proposal, fork the environment and simulate both actions for one step:

```
regret = reward(proposed_action) - reward(filtered_action)
```

- **Near-zero regret** → agent already proposes safe actions (filter is redundant)
- **Large positive regret** → filter prevents real harm (filter is valuable)
- **Large negative regret** → filter makes things worse (filter miscalibrated)

---

## 10. Project Structure

```
ContainerScale-RL/
├── src/
│   ├── env/
│   │   ├── k8s_sim.py              # Gymnasium environment
│   │   └── workload.py             # 5 traffic patterns
│   ├── models/                      # Experiment 3 only
│   │   ├── structured_model.py
│   │   ├── flat_model.py
│   │   └── ensemble.py
│   ├── training/
│   │   └── train_ppo.py            # RecurrentPPO (standard SB3)
│   ├── safety/
│   │   └── safety_filter.py
│   ├── agents/
│   │   ├── hpa_baseline.py         # RealisticHPA
│   │   └── agent.py                # ContainerScaleAgent
│   ├── live/                        # Live cluster deployment
│   │   ├── observer.py             # PrometheusObserver
│   │   ├── executor.py             # K8sPatchExecutor
│   │   ├── live_agent.py           # LiveClusterAgent
│   │   └── metrics_logger.py
│   └── evaluation/
│       ├── sim_experiments.py       # Experiments 1-7
│       ├── live_experiment.py       # Experiment 8
│       ├── regret_analysis.py
│       ├── metrics.py
│       └── visualize.py
├── deploy/                          # K8s manifests
│   ├── k3s-setup.sh
│   ├── podinfo.yaml
│   ├── prometheus.yaml
│   ├── hpa.yaml
│   └── locust/
│       ├── steady.py
│       ├── diurnal.py
│       ├── flash_crowd.py
│       └── gradual_ramp.py
├── notebooks/
│   ├── 01_simulator_validation.ipynb
│   ├── 02_world_model_comparison.ipynb
│   ├── 03_sim_results.ipynb
│   └── 04_live_comparison.ipynb
└── configs/
    ├── env_config.yaml
    └── training_config.yaml
```

---

## 11. Dependencies

```
# Simulation & Training
gymnasium>=0.29
sb3-contrib>=2.0          # RecurrentPPO
stable-baselines3>=2.0
torch>=2.0
numpy
matplotlib
tensorboard

# Live Cluster
kubernetes>=28.0
prometheus-api-client>=0.5
locust>=2.20
pyyaml
```

---

## 12. Known Limitations

1. **±3 replica action space.** Cannot handle arbitrary step-function spikes. Flash crowd capped at 450 rps.
2. **Single deployment.** Multi-service joint scaling is out of scope.
3. **Pod scaling only.** Node provisioning handled by Cluster Autoscaler (not controlled).
4. **Sim-to-real gap.** Domain randomization mitigates but doesn't eliminate transfer error.
5. **Local k3s ≠ production cloud.** Network, storage, and scheduling differ from multi-node production clusters.
6. **Locust workloads are synthetic.** Real traffic has different characteristics.
7. **LSTM anticipation is implicit.** The agent learns temporal patterns from training data; it is not a forecaster with explicit traffic prediction.
8. **Reward weights are hyperparameters.** Experiment 5 provides the Pareto frontier for tuning.
