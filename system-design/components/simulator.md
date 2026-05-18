# Component: Simulator (K8sSimEnv)

## Responsibility

Simulates a Kubernetes cluster environment for RL agent training. Implements the Gymnasium `Env` interface so any compatible RL algorithm can train against it without custom glue code.

---

## Interface

```
reset()         → state: ndarray[22]
step(action)    → (state: ndarray[22], reward: float, done: bool, info: dict)
```

---

## Inputs and Outputs

| Method | Input | Output |
|---|---|---|
| `reset()` | None | Initial 22-dim observation vector |
| `step(action)` | Integer 0–6 (maps to replica delta -3 to +3) | (new state, reward, done, info) |

The action integer mapping:

| Action | Replica Delta |
|---|---|
| 0 | -3 |
| 1 | -2 |
| 2 | -1 |
| 3 | 0 (no change) |
| 4 | +1 |
| 5 | +2 |
| 6 | +3 |

---

## Internal State

Variables the simulator tracks and updates between steps:

| Variable | Description |
|---|---|
| `replicas` | Current number of ready pods |
| `pending_pods` | List of pods still starting up, each with remaining startup time |
| `queue_depth` | Accumulated request backlog (requests waiting to be served) |
| `request_rate` | Current traffic in requests per second |
| `p99_latency` | Current tail latency in milliseconds |
| `cpu_util` | Current CPU utilization across pods |
| `mem_util` | Current memory utilization across pods |
| `cost_rate` | Current cost in $/hour |
| `step_count` | Current step within the episode (0–119) |
| `per_pod_capacity` | How many rps each pod can handle (randomized per episode) |
| `cold_start_mean` | Average pod startup time in seconds (randomized per episode) |
| `node_price` | Cost per node per hour (randomized per episode) |

---

## Observation Vector (22 dimensions)

What the agent actually sees — derived from internal state:

```
[0]  cpu_util
[1]  mem_util
[2]  replicas / max_replicas          (normalized)
[3]  pending_pods / 10                (normalized)
[4]  request_rate / 500               (normalized)
[5]  (request_rate - prev_rate) / 100 (traffic trend/derivative)
[6]  p99_latency / 1000               (normalized)
[7]  per_pod_capacity / 30            (normalized)
[8]  queue_depth / 10000              (normalized)
[9]  node_0_cpu_available / cpu_total
[10] node_0_mem_available / mem_total
[11] node_0_pod_count / 30
[12] node_1_cpu_available / cpu_total
[13] node_1_mem_available / mem_total
[14] node_1_pod_count / 30
[15] node_2_cpu_available / cpu_total
[16] node_2_mem_available / mem_total
[17] node_2_pod_count / 30
[18] sin(2π × step / 120)             (time encoding)
[19] cos(2π × step / 120)             (time encoding)
[20] cost_rate / 2.0                  (normalized)
[21] prev_cpu_util                    (previous step CPU for trend)
```

---

## Internal Logic (Per Step)

Each call to `step(action)` runs the following in order:

### 1. Apply Scaling Action
- Convert action integer to replica delta (-3 to +3)
- Scale up: create new pending pods, each with startup time sampled from LogNormal(mean=cold_start_mean, σ=0.5), clipped to [30s, 180s]
- Scale down: cancel pending pods first (cheaper), then remove ready pods
- Enforce min=2, max=30 replica bounds

### 2. Age Pending Pods
- Each pending pod's remaining startup time decreases by 30 seconds (one step)
- Pods with remaining time ≤ 0 graduate to ready

### 3. Update Traffic
- Ask WorkloadGenerator for current request rate at this step
- Compute traffic derivative (change from previous step)

### 4. Queue Dynamics (Little's Law)
```
service_rate     = ready_pods × per_pod_capacity
arrivals         = request_rate × 30
departures       = service_rate × 30
queue_depth      = max(0, queue_depth + arrivals - departures)
queue_wait       = (queue_depth / request_rate) × 1000   [ms]
p99_latency      = base_latency + queue_wait + noise
```

### 5. Compute Cost
```
cpu_per_pod   = 0.25 cores
nodes_needed  = ceil(total_pod_cpu / (node_cpu × 0.85))
cost_rate     = nodes_needed × node_price_per_hour
```

### 6. Compute Reward
```
r_sla       = -10.0 × max(0, (p99_latency - sla_target) / sla_target)
r_cost      = -0.1 × cost_rate
r_crash     = -50.0 if p99_latency > 5 × sla_target else 0
r_stability = -0.3 × |delta|  only if p99_latency < 0.5 × sla_target
                               (no stability penalty during overload)

reward = r_sla + r_cost + r_crash + r_stability
```

### 7. Build Observation Vector
- Derive all 22 values from updated internal state
- Normalize each value to roughly [0, 1] range

### 8. Check Episode End
- `done = True` if step_count >= 120 (1 simulated hour)

---

## Domain Randomization (on reset)

Each episode randomizes these parameters so the agent learns general behavior, not memorized responses to one fixed configuration:

| Parameter | Range | Why |
|---|---|---|
| `per_pod_capacity` | 15–30 rps/pod | Agent must adapt to different pod performance |
| `cold_start_mean` | 40–100 seconds | Agent must handle variable startup delays |
| `node_price` | $0.20–$0.50/hour | Agent must generalize cost tradeoffs |
| `workload_pattern` | One of 5 patterns | Agent must handle all traffic shapes |

---

## Workload Patterns

Five traffic shapes, randomly selected per episode:

| Pattern | Shape | Peak RPS |
|---|---|---|
| Steady | Flat ~100 rps + Gaussian noise | 100 |
| Diurnal | Sine wave, 50→200→50 | 200 |
| Flash crowd | Baseline 100, spike to 450 at t=0.3 | 450 |
| Gradual ramp | Linear 50→450 over episode | 450 |
| Noisy | Log-normal (mean=100, σ=0.7) | Variable |

Flash crowd peak is capped at 450 rps because at minimum per_pod_capacity (15 rps/pod), the agent needs exactly 30 replicas — the maximum allowed. Every workload is physically serviceable within action bounds.

---

## Failure Modes

| Failure | What Happens | Mitigation |
|---|---|---|
| Queue overflow | queue_depth grows unbounded if severely under-provisioned | Reward crash penalty kicks in at 5× SLA target |
| Reward miscalibration | Wrong weight balance causes agent to over-optimize one objective | Experiment 5 sweeps cost weight α to find Pareto frontier |
| Unrealistic physics | Simulator doesn't perfectly model real cluster behavior | Domain randomization + sim-to-real gap analysis in Experiment 8 |
| Division by zero | request_rate = 0 causes queue_wait calculation to fail | Clamp request_rate to minimum value on each step |

---

## Dependencies

| Dependency | Purpose |
|---|---|
| `gymnasium` | Base `Env` class and space definitions |
| `numpy` | State vector construction, math operations |
| `WorkloadGenerator` | Provides request rate at each step |
| `stable-baselines3` | Training loop that calls `reset()` and `step()` |

---

## Performance Requirement

Must sustain **1000+ steps/second** to make 500,000-step training feasible in reasonable wall-clock time. All internal logic must be vectorized (NumPy) — no Python loops over pods or nodes.
