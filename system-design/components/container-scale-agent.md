# Component: ContainerScaleAgent

## Responsibility

Loads the trained RecurrentPPO model and runs inference at each control loop step. Takes the 22-dim observation vector, maintains LSTM hidden state across steps, and produces a replica delta decision. Falls back to HPA on any failure.

---

## Inputs and Outputs

| | Description |
|---|---|
| **Input** | 22-dim observation vector (from PrometheusObserver) |
| **Input** | Current step number (for safety filter) |
| **Output** | Replica delta integer (-3 to +3) |

---

## Interface

```
__init__()                          → ContainerScaleAgent
decide(state, step)                 → int  (replica delta)
reset()                             → None (clears LSTM hidden state)
```

---

## Internal Logic

### `decide(state, step)`

```
1. Try:
    a. Run state through RecurrentPPO with current LSTM hidden state
    b. Get action (0–6) and updated LSTM hidden state
    c. Map action to delta: delta = action - 3
    d. Pass delta through SafetyFilter
    e. Return safe delta

2. On any exception:
    a. Fall back to RealisticHPA.act(state)
    b. Return HPA delta
```

### `reset()`
Clears LSTM hidden state to None. Must be called at the start of each new episode or live run so the agent doesn't carry memory from a previous session.

---

## The LSTM Hidden State

This is the agent's working memory. It is a pair of tensors (hidden state + cell state) that persist between calls to `decide()`.

At step 0: `lstm_states = None` (SB3 initializes automatically)
At step N: `lstm_states` contains a summary of everything the agent has observed in steps 0 through N-1

This is what enables anticipation — the agent's decision at step 60 is informed by everything it saw in steps 0–59, not just the current observation.

**Critical:** If `lstm_states` is not reset between episodes, the agent carries stale memory from the previous run into the new one, causing incorrect behavior.

---

## How the Neural Network Produces a Decision

The policy network runs the following internally on each call:

```
Input: 22-dim observation vector
         ↓
    Linear layer (256 neurons)
         ↓
    Linear layer (256 neurons)
         ↓
    LSTM layer (128 hidden units)
    (takes previous hidden state, outputs new hidden state)
         ↓
    Output layer (7 neurons)
         ↓
    Softmax → 7 probabilities
    [0.02, 0.05, 0.08, 0.15, 0.40, 0.25, 0.05]
      -3    -2    -1    0    +1    +2    +3
         ↓
    Argmax (deterministic mode) → single integer (e.g., 4 = "+1")
```

The output is not text. It is a single integer representing which of the 7 possible actions to take.

---

## Fallback Behavior

On any exception during inference (model file missing, tensor shape mismatch, CUDA error, etc.):

1. Log the error
2. Call `RealisticHPA.act(state)` instead
3. Return HPA's delta
4. Continue — do not crash the control loop

The system never stops making scaling decisions. It degrades gracefully from RL to HPA rather than failing open (no scaling) or failing closed (crash).

---

## Internal State

| Variable | Description |
|---|---|
| `ppo` | Loaded RecurrentPPO model (frozen weights) |
| `lstm_states` | Current LSTM hidden state (updated each step) |
| `safety` | SafetyFilter instance |
| `hpa` | RealisticHPA instance (fallback) |

---

## Failure Modes

| Failure | What Happens | Mitigation |
|---|---|---|
| Model file not found | Exception on load | Fail fast at startup with clear error message |
| LSTM state corruption | Garbage output | Reset lstm_states, fall back to HPA for that step |
| Inference too slow | Control loop misses 30s window | Log warning; inference should be <100ms |
| Action out of bounds | Safety filter catches it | SafetyFilter clips to valid range |

---

## Dependencies

| Dependency | Purpose |
|---|---|
| `sb3_contrib.RecurrentPPO` | Model loading and inference |
| `SafetyFilter` | Validates every action before returning |
| `RealisticHPA` | Fallback controller |
| `numpy` | State vector handling |
