# Component: SafetyFilter

## Responsibility

Enforces hard-coded invariant rules on every scaling action before it reaches the Kubernetes API. Never trained. Never overridden by the agent. Acts as the last line of defense between the RL agent's decision and the real cluster.

---

## Inputs and Outputs

| | Description |
|---|---|
| **Input** | Current cluster state (observation vector) |
| **Input** | Proposed replica delta from RL agent (-3 to +3) |
| **Input** | Current step number |
| **Output** | Safe replica delta (may be modified or zeroed) |

---

## Interface

```
__init__()                                    → SafetyFilter
check(state, proposed_delta, step)            → int  (safe delta)
reset()                                       → None (resets cooldown timer)
```

---

## The Four Rules

Applied in order on every action:

### Rule 1: Absolute Replica Bounds
```
new_replicas = state.replicas + proposed_delta
if new_replicas < 2:  proposed_delta = 2 - state.replicas
if new_replicas > 30: proposed_delta = 30 - state.replicas
```
The cluster must always have between 2 and 30 replicas. No exceptions.

**Why 2 minimum:** Single replica means one pod failure takes the service down entirely.
**Why 30 maximum:** Node capacity limit — beyond this, pods can't be scheduled.

### Rule 2: No Scale-Down During High Latency
```
if proposed_delta < 0 and state.p99_latency > 0.8 × sla_target:
    proposed_delta = 0
```
If the system is already struggling (latency above 80% of SLA target), scaling down would make it worse. Block it unconditionally.

### Rule 3: Maximum Delta Per Step
```
proposed_delta = clip(proposed_delta, -3, +3)
```
Prevents any single step from making a large jump. Limits blast radius of a bad decision.

### Rule 4: Rate Limiting (Cooldown)
```
if (step - last_action_step) < 1 and proposed_delta != 0:
    proposed_delta = 0
```
Enforces a minimum of one step (30 seconds) between scaling actions. Prevents rapid oscillation (scale up, scale down, scale up in quick succession).

---

## Why These Rules Are Hard-Coded

Early in training, PPO proposes wild actions — scale to 0, scale by 10, scale down during a spike. Without the safety filter, these actions would cause catastrophic simulator states that corrupt the training signal.

As training converges, the agent learns that proposals outside these bounds get clipped anyway, so it stops making them. Experiment 4 measures this directly — tracking how often the filter actually modifies the agent's proposal over the course of training. The expectation is that filter interventions start high and converge toward zero.

**The filter is never removed** — even in live deployment. A trained agent that has internalized safe behavior still passes through the filter. The cost is near-zero (a few microseconds). The benefit is a hard guarantee that no software bug in the agent can cause a cluster outage.

---

## Regret Analysis (Experiment 4)

For each step where the filter modifies the agent's proposal, the experiment forks the environment and simulates both actions:

```
regret = reward(proposed_action) - reward(filtered_action)
```

| Regret Value | Interpretation |
|---|---|
| Near zero | Agent already proposes safe actions — filter is redundant but harmless |
| Large positive | Filter prevents real harm — filter is valuable |
| Large negative | Filter makes things worse — filter may be miscalibrated |

---

## Internal State

| Variable | Description |
|---|---|
| `last_action_step` | Step number of the last non-zero scaling action |
| `MIN_REPLICAS` | 2 (constant) |
| `MAX_REPLICAS` | 30 (constant) |
| `MAX_DELTA` | 3 (constant) |
| `MIN_INTERVAL_STEPS` | 1 (constant) |

---

## Failure Modes

| Failure | What Happens | Mitigation |
|---|---|---|
| State vector malformed | Rule 2 reads garbage latency value | Validate state before passing to filter |
| All rules block valid action | Agent can't respond to emergency | Rule 2 only blocks scale-DOWN, never scale-up |
| Cooldown too aggressive | Agent can't respond fast enough | MIN_INTERVAL_STEPS=1 means 30s cooldown — tunable |

---

## Dependencies

None. The SafetyFilter is intentionally dependency-free — pure Python with no external libraries. This minimizes the chance of a bug or import failure disabling the safety layer.
