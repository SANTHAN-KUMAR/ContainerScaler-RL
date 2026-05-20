# Component: RealisticHPA (HPA Baseline)

## Responsibility

Implements a faithful replica of Kubernetes' Horizontal Pod Autoscaler behavior, including the 3-minute scale-down stabilization window. Serves two roles: the comparison baseline in experiments, and the fallback controller when the RL agent fails.

---

## Inputs and Outputs

| | Description |
|---|---|
| **Input** | Current cluster state (observation vector) |
| **Output** | Replica delta integer |

---

## Interface

```
__init__(target_cpu=0.50)    → RealisticHPA
act(state)                   → int  (replica delta)
reset()                      → None (clears stabilization window history)
```

---

## Internal Logic

### The HPA Formula
```
desired_replicas = ceil(current_replicas × cpu_util / target_cpu)
desired_replicas = clip(desired_replicas, 2, 30)
```

This is the actual formula Kubernetes HPA uses. At 50% CPU target:
- If 4 replicas are running at 80% CPU → desired = ceil(4 × 0.80 / 0.50) = ceil(6.4) = 7
- If 4 replicas are running at 30% CPU → desired = ceil(4 × 0.30 / 0.50) = ceil(2.4) = 3

### Scale-Down Stabilization Window
Real Kubernetes HPA does not scale down immediately when CPU drops. It maintains a 3-minute history of desired replica counts and uses the maximum:

```
replica_history.append(desired_replicas)   # deque of last 6 values (6 × 30s = 3 min)

if desired_replicas < current_replicas:
    desired_replicas = max(replica_history)  # use highest recent value
```

This prevents thrashing — if CPU briefly dips, HPA waits to confirm the drop is sustained before scaling down.

**This is why the baseline is "realistic"** — a naive HPA without this window would scale down too aggressively, making the comparison unfair to HPA.

---

## Why This Matters for the Comparison

The RL agent is compared against this baseline, not a strawman. If the baseline were naive (no stabilization window), any improvement from RL could be attributed to the baseline being unrealistically bad rather than the RL agent being genuinely better.

By matching real Kubernetes behavior, any performance difference in the experiments is attributable to the fundamental limitations of the reactive proportional control approach, not implementation quality.

---

## HPA's Structural Limitations (What It Cannot Do)

| Limitation | Why |
|---|---|
| Cannot anticipate traffic | Only reacts to current CPU — no memory, no trend |
| Cannot account for cold-start | Assumes new pods serve instantly |
| Cannot optimize cost | Only targets CPU utilization percentage |
| Scales down slowly | 3-minute stabilization window is conservative by design |

These are the exact limitations the RL agent is designed to overcome.

---

## Internal State

| Variable | Description |
|---|---|
| `target_cpu` | CPU utilization target (default 0.50 = 50%) |
| `replica_history` | Deque of last 6 desired replica counts (3-minute window) |

---

## Failure Modes

| Failure | What Happens | Mitigation |
|---|---|---|
| CPU util = 0 | Formula produces 0 desired replicas | Clip to minimum 2 replicas |
| CPU util very high | Formula produces > 30 replicas | Clip to maximum 30 replicas |
| History empty | max() on empty deque fails | Initialize history with current replica count |

---

## Role as Fallback

When `ContainerScaleAgent` catches any exception during RL inference, it calls `hpa.act(state)` instead. This means:

- The cluster is never left without a controller
- The fallback uses the same state vector format as the RL agent
- No special handling needed — HPA reads the same observation the RL agent would have used

---

## Dependencies

| Dependency | Purpose |
|---|---|
| `collections.deque` | Fixed-size stabilization window history |
| `math.ceil` | HPA formula ceiling function |
