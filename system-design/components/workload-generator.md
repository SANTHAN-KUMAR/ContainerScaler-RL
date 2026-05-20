# Component: WorkloadGenerator

## Responsibility

Generates synthetic traffic (request rate in rps) at each simulation step, mimicking real-world traffic patterns. Called by the Simulator on every step to get the current request rate.

---

## Interface

```
__init__(pattern: str)     → WorkloadGenerator
get_rate(step: int)        → float  (requests per second)
reset()                    → None   (pick new random pattern)
```

---

## Inputs and Outputs

| Method | Input | Output |
|---|---|---|
| `__init__` | pattern name (or "random") | WorkloadGenerator instance |
| `get_rate()` | current step number (0–119) | request rate in rps (float) |
| `reset()` | None | None — updates internal pattern selection |

---

## Internal State

| Variable | Description |
|---|---|
| `pattern` | Which traffic shape is active this episode |
| `noise_seed` | Random seed for reproducible noise generation |
| `peak_rps` | Maximum request rate for this episode's pattern |

---

## Traffic Patterns

Five shapes, one randomly selected per episode:

### 1. Steady
```
rate = 100 + Gaussian(mean=0, σ=10)
```
Flat baseline with small random noise. Tests basic cost optimization.

### 2. Diurnal
```
rate = 125 + 75 × sin(2π × step / 120)
range: 50 → 200 → 50 rps
```
Sine wave simulating daily traffic cycle. Tests whether LSTM learns time-of-day patterns.

### 3. Flash Crowd
```
step < 36:   rate = 100 + noise
step 36–48:  rate = ramps up to 450
step 48–72:  rate = 450 + noise
step > 72:   rate = decays back to 100
```
Sudden spike at ~30% into the episode. Tests cold-start pre-scaling behavior.

### 4. Gradual Ramp
```
rate = 50 + (400 × step / 119) + noise
range: 50 → 450 rps linearly
```
Steady linear increase over the full episode. Tests sustained anticipation.

### 5. Noisy
```
rate = LogNormal(mean=100, σ=0.7)
```
Highly variable, unpredictable traffic. Tests robustness to chaos.

---

## Why These Patterns

| Pattern | What It Tests |
|---|---|
| Steady | Cost efficiency — no excuse to over-provision |
| Diurnal | Temporal pattern learning via LSTM memory |
| Flash crowd | Cold-start anticipation — must pre-scale before spike |
| Gradual ramp | Sustained trend following |
| Noisy | Robustness — don't over-react to noise |

Flash crowd peak is capped at 450 rps because at minimum per_pod_capacity (15 rps/pod), the agent needs exactly 30 replicas — the maximum allowed. Every pattern is physically serviceable within the action space.

---

## Failure Modes

| Failure | What Happens | Mitigation |
|---|---|---|
| Rate goes negative | Gaussian noise on low baseline could produce negative rps | Clamp output to minimum of 1.0 rps |
| Pattern not diverse enough | Agent memorizes patterns instead of generalizing | Domain randomization — pattern randomly selected each episode |
| Noise too high | Agent can't distinguish signal from noise | σ values tuned so signal-to-noise ratio is learnable |

---

## Dependencies

| Dependency | Purpose |
|---|---|
| `numpy` | Random number generation, math functions |

---

## Relationship to Simulator

The WorkloadGenerator is owned and called by the Simulator. On each `step()`, the Simulator calls `get_rate(current_step)` to get the current traffic level, which then drives the queue dynamics calculation.

On each `reset()`, the Simulator calls `workload.reset()` to randomly select a new pattern for the next episode.

```
Simulator.step()
    → workload.get_rate(step)
    → returns rps float
    → used in queue dynamics: arrivals = rps × 30
```
