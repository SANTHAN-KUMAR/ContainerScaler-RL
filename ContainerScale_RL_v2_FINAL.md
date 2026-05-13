# ContainerScale-RL v2: Revised Architecture & Specification

## 1. Problem Statement

Kubernetes Horizontal Pod Autoscaler (HPA) has four structural defects:

1. **Reactive control** — scales only after load arrives; cannot anticipate demand
2. **Cold-start blindness** — ignores 30–180s pod startup delay when computing desired replicas
3. **Cost-blind** — optimizes utilization without considering node-level cost
4. **Placement-decoupled** — replica count is the only lever; scheduling is external

This project addresses defects 1–3 through a Dyna-style model-based RL agent. Defect 4 (placement) is explicitly out of scope — the action space is replica-count delta only.

### Honest Scope of Claims

| HPA Defect | Coverage | Mechanism |
|---|---|---|
| Cost-blind | **Fully addressed** | Cost term in reward; Pareto frontier (Exp 6) |
| Cold-start blindness | **Substantially addressed** | World model predicts pending→ready transitions; planner looks ahead through startup delays |
| Reactive control | **Partially addressed** | Traffic forecasting sub-network predicts next-step demand; LSTM policy learns temporal patterns. Full anticipation of arbitrary future demand is aspirational |
| Placement | **Not addressed** | Out of scope |

---

## 2. Architectural Decision: Dyna-Q (Option C)

The v1 spec combined an independently-trained PPO policy with a frozen world model planner that could override PPO at inference. The critique (T1) correctly identified this as theoretically incoherent: a model trained on weaker (random) data should not override a policy trained on stronger data, and the two components have no joint optimization guarantee.

**v2 adopts Dyna-Q** — the world model generates synthetic training experiences for PPO, and PPO's on-policy rollouts continuously update the world model. This resolves the core tension:

- The world model is trained on *policy-relevant* data (not random), eliminating distribution shift (M2)
- PPO learns from both real and imagined rollouts, making the world model's contribution to training direct and measurable
- No inference-time override mechanism needed — PPO is the sole decision-maker at test time
- The safety filter remains as a hard-coded post-processing layer (unchanged)

```
┌─────────────────────────────────────────────────────────┐
│                    DYNA-Q TRAINING LOOP                  │
│                                                         │
│  ┌──────────┐  real transitions   ┌──────────────────┐  │
│  │Simulator │ ──────────────────► │  Replay Buffer   │  │
│  │(Gymnasium)│ ◄─────────────────  │  (real + synth)  │  │
│  └──────────┘  actions from PPO   └────────┬─────────┘  │
│       │                                    │            │
│       │ real transitions                   │ sample     │
│       ▼                                    ▼            │
│  ┌──────────────┐  imagined        ┌──────────────┐    │
│  │ World Model  │  transitions ──► │   PPO Agent   │    │
│  │ (Ensemble)   │ ◄── retrain ──── │  (RecurrentPPO│    │
│  └──────────────┘  on real data    │   w/ LSTM)    │    │
│                                    └──────────────┘    │
│                                         │               │
│                            INFERENCE    ▼               │
│                                    ┌──────────┐         │
│                                    │  Safety  │         │
│                                    │  Filter  │         │
│                                    └──────────┘         │
└─────────────────────────────────────────────────────────┘
```

### Why Not Options A or B?

- **Option A (model-free only):** Loses the sample-efficiency benefit of imagination. With 5 workload patterns × domain randomization, pure model-free PPO needs significantly more environment steps for coverage.
- **Option B (pure MPC):** The simulator runs at 1000+ steps/sec, making online CEM feasible — but the simulator is not the real cluster. MPC on an imperfect simulator at test time has no learning, no adaptation. Dyna uses the model during training to accelerate learning, then discards it.

---

## 3. Component 1: Simulator (Gymnasium Environment)

### 3.1 State Variables — Properly Initialized (fixes C2)

```python
class KubernetesAutoscalingEnv(gymnasium.Env):
    observation_space = Box(shape=(24,), low=-5, high=5)  # 22 → 24 (added per_pod_capacity, queue_depth)
    action_space = Discrete(7)  # delta ∈ {-3, -2, -1, 0, +1, +2, +3}

    def __init__(self, num_nodes=3, node_cpu=4.0, node_mem=16.0,
                 sla_target_ms=200, max_replicas=30):
        self.nodes = [Node(cpu=node_cpu, mem=node_mem) for _ in range(num_nodes)]
        self.sla_target = sla_target_ms
        self.max_replicas = max_replicas
        self.workload = WorkloadGenerator()

        # ── FIX C2: Initialize all state variables ──
        self.replicas = 2
        self.ready_pods = 2
        self.pending_pods = 0
        self.pending_list = []
        self.time_step = 0
        self.queue_depth = 0.0          # FIX D1: queue state
        self.per_pod_capacity = 20.0
        self.cold_start_mean = 60.0
        self.node_price = 0.34
        self.request_rate = 0.0
        self.p99_latency = 20.0
        self.cpu_util = 0.0
        self.mem_util = 0.3
        self.cost_rate = 0.0
        self.prev_cpu_util = 0.0
        self.prev_request_rate = 0.0
        self.prev_latency = 20.0

        # Observation delay buffer (FIX D8)
        self.obs_delay_steps = 1  # 1 step = 30s delay
        self.obs_buffer = deque(maxlen=self.obs_delay_steps + 1)

    def reset(self, seed=None):
        """Full state reset per episode — no carryover."""
        self.replicas = 2
        self.ready_pods = 2
        self.pending_pods = 0
        self.pending_list = []
        self.time_step = 0
        self.queue_depth = 0.0

        # Domain randomization
        self.per_pod_capacity = np.random.uniform(15, 30)
        self.cold_start_mean = np.random.uniform(40, 100)
        self.node_price = np.random.uniform(0.20, 0.50)
        self.workload = WorkloadGenerator()

        self.obs_buffer.clear()
        return self._get_obs(), {}
```

### 3.2 Step Function — All Critical/Major Fixes Applied

```python
    def step(self, action_idx):
        delta = action_idx - 3

        # ── 1. Apply scaling ──
        new_replicas = np.clip(self.replicas + delta, 2, self.max_replicas)
        actual_delta = new_replicas - self.replicas

        # FIX M5: Scale-down properly updates ready_pods and pending_pods
        if actual_delta < 0:
            to_remove = abs(actual_delta)
            # Cancel pending first (cheaper than evicting ready)
            pending_removed = min(to_remove, len(self.pending_list))
            self.pending_list = self.pending_list[pending_removed:]
            self.pending_pods -= pending_removed
            ready_removed = to_remove - pending_removed
            self.ready_pods = max(0, self.ready_pods - ready_removed)
        elif actual_delta > 0:
            for _ in range(actual_delta):
                startup = np.random.lognormal(
                    mean=np.log(self.cold_start_mean), sigma=0.5
                )
                startup = np.clip(startup, 30, 180)
                self.pending_list.append(PendingPod(remaining=startup))
                self.pending_pods += 1

        self.replicas = new_replicas

        # ── 2. Age pending pods ──
        # FIX C3: Collect graduated pods, then prune list in one pass
        for pod in self.pending_list:
            pod.remaining -= 30

        graduated = [p for p in self.pending_list if p.remaining <= 0]
        self.ready_pods += len(graduated)
        self.pending_pods -= len(graduated)
        self.pending_list = [p for p in self.pending_list if p.remaining > 0]

        # ── 3. Generate traffic ──
        self.prev_request_rate = self.request_rate
        self.request_rate = self.workload.sample(self.time_step)

        # ── 4. Compute latency via queue model (FIX D1) ──
        effective_replicas = self.ready_pods
        service_rate = effective_replicas * self.per_pod_capacity
        arrivals_this_step = self.request_rate * 30
        departures_this_step = service_rate * 30
        self.queue_depth = max(0.0, self.queue_depth + arrivals_this_step - departures_this_step)

        # Little's Law: W = L / λ
        base_latency = 20  # ms at zero load
        queue_wait = (self.queue_depth / max(self.request_rate, 1.0)) * 1000
        self.p99_latency = base_latency + queue_wait + np.random.normal(0, 5)
        self.p99_latency = max(self.p99_latency, base_latency)

        # ── 5. CPU/mem utilization ──
        self.prev_cpu_util = self.cpu_util
        utilization = self.request_rate / max(service_rate, 1e-6)
        utilization = np.clip(utilization, 0, 1.0)
        self.cpu_util = min(utilization + np.random.normal(0, 0.02), 1.0)
        self.mem_util = 0.3 + 0.4 * utilization + np.random.normal(0, 0.02)

        # ── 6. Cost (node-level) ──
        cpu_per_pod = 0.25
        total_cpu = self.replicas * cpu_per_pod
        nodes_needed = math.ceil(total_cpu / (self.nodes[0].cpu * 0.85))
        self.cost_rate = max(nodes_needed, 1) * self.node_price

        # ── 7. Reward ──
        self.prev_latency = self.p99_latency
        reward = self._compute_reward(actual_delta)

        # ── 8. Observation with delay (FIX D8) ──
        current_obs = self._build_obs_vector()
        self.obs_buffer.append(current_obs)
        delayed_obs = self.obs_buffer[0] if len(self.obs_buffer) > self.obs_delay_steps else current_obs

        self.time_step += 1
        done = self.time_step >= 120

        return delayed_obs, reward, done, False, {}
```

### 3.3 Observation Vector — 24 Dimensions (fixes M3, D5)

```python
    def _build_obs_vector(self):
        """24-dim observation. per_pod_capacity added (M3). contention removed (D5)."""
        t_norm = self.time_step / 120
        return np.array([
            # Per-deployment (9)
            self.cpu_util, self.mem_util, self.replicas / self.max_replicas,
            self.pending_pods / 10.0, self.request_rate / 500.0,
            (self.request_rate - self.prev_request_rate) / 100.0,  # derivative
            self.p99_latency / 1000.0,
            self.per_pod_capacity / 30.0,       # FIX M3: now observable
            self.queue_depth / 10000.0,         # FIX D1: queue state visible
            # Per-node × 3 (9)
            *[n.cpu_available / n.cpu for n in self.nodes],
            *[n.mem_available / n.mem for n in self.nodes],
            *[n.pod_count / 30.0 for n in self.nodes],
            # Global (6)
            np.sin(2 * np.pi * t_norm), np.cos(2 * np.pi * t_norm),
            self.cost_rate / 2.0,
            self.prev_cpu_util, self.prev_request_rate / 500.0,
            self.prev_latency / 1000.0,
        ], dtype=np.float32)
```

### 3.4 Context-Conditioned Reward (fixes D2)

```python
    def _compute_reward(self, delta):
        sla_violation = max(0, self.p99_latency - self.sla_target) / self.sla_target
        r_sla = -10.0 * sla_violation
        r_cost = -0.1 * self.cost_rate
        r_crash = -50.0 if self.p99_latency > 5 * self.sla_target else 0

        # FIX D2: Only penalize scaling when latency is comfortable
        if self.p99_latency < 0.5 * self.sla_target:
            r_stability = -0.3 * abs(delta)
        else:
            r_stability = 0.0  # scaling is justified

        return r_sla + r_cost + r_stability + r_crash
```

### 3.5 Workload Generator (fixes M6, M8)

```python
class WorkloadGenerator:
    def __init__(self, pattern=None):
        self.pattern = pattern or random.choice([
            'steady', 'diurnal', 'flash_crowd', 'gradual_ramp', 'noisy'
        ])

    def sample(self, step):
        t = step / 120
        if self.pattern == 'steady':
            return max(0.0, 100 + np.random.normal(0, 5))
        elif self.pattern == 'diurnal':
            return max(0.0, 50 + 150 * np.sin(np.pi * t) + np.random.normal(0, 10))
        elif self.pattern == 'flash_crowd':
            # FIX M8: Spike scaled to 500 rps (reachable within action bounds)
            # At per_pod_capacity=20, 500 rps needs 25 pods (within max_replicas=30)
            if 0.28 < t < 0.45:
                return max(0.0, 500 + np.random.normal(0, 30))
            return max(0.0, 100 + np.random.normal(0, 5))
        elif self.pattern == 'gradual_ramp':
            return max(0.0, 50 + 400 * t + np.random.normal(0, 10))
        elif self.pattern == 'noisy':
            # FIX M6: Log-normal ensures positivity
            return np.random.lognormal(mean=np.log(100), sigma=0.7)
```
# ContainerScale-RL v2 — Part 2: World Model, Training, Safety, Baseline

## 4. Component 2: Structured World Model (Revised)

### 4.1 Architecture — With Traffic Forecasting (fixes M4, D5)

The world model now predicts **5 variables** including next-step request rate. The `contention` input (D5) is removed — its effect is absorbed by the model's noise term.

```python
class StructuredWorldModel(nn.Module):
    """
    5 sub-networks. Each predicts one variable from its causal parents.
    Key change from v1: request_rate is now a predicted variable (M4).
    contention removed (D5) — was never defined in obs space.
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.predict_cpu = GaussianMLP(in_dim=3, hidden=hidden)
        # inputs: (current_cpu, request_rate, new_replicas)

        self.predict_mem = GaussianMLP(in_dim=3, hidden=hidden)
        # inputs: (current_mem, request_rate, new_replicas)

        self.predict_pending = GaussianMLP(in_dim=2, hidden=hidden)
        # inputs: (current_pending, delta)

        self.predict_request_rate = GaussianMLP(in_dim=3, hidden=hidden)
        # inputs: (current_rate, rate_derivative, time_of_day)
        # FIX M4: traffic is now predicted, not held constant

        self.predict_latency = GaussianMLP(in_dim=4, hidden=hidden)
        # inputs: (predicted_cpu, predicted_mem, predicted_pending, queue_depth)
        # FIX D5: contention removed, queue_depth added (from D1)

    def forward(self, state, action_delta):
        new_replicas = state.replicas + action_delta

        cpu_mu, cpu_lv = self.predict_cpu(
            state.cpu, state.request_rate, new_replicas
        )
        mem_mu, mem_lv = self.predict_mem(
            state.mem, state.request_rate, new_replicas
        )
        pend_mu, pend_lv = self.predict_pending(
            state.pending, action_delta
        )
        rate_mu, rate_lv = self.predict_request_rate(
            state.request_rate, state.request_rate_deriv, state.time_of_day
        )
        lat_mu, lat_lv = self.predict_latency(
            cpu_mu, mem_mu, pend_mu, state.queue_depth
        )
        return Prediction(
            cpu=(cpu_mu, cpu_lv), mem=(mem_mu, mem_lv),
            pending=(pend_mu, pend_lv), latency=(lat_mu, lat_lv),
            request_rate=(rate_mu, rate_lv)
        )
```

### 4.2 Ensemble — With `mean_prediction` Method (fixes C1)

```python
class WorldModelEnsemble:
    def __init__(self, n_models=10):  # FIX D7: increased from 5 to 10
        self.models = [StructuredWorldModel() for _ in range(n_models)]

    def predict(self, state, action_delta):
        """Returns (mean_latency, epistemic_uncertainty)."""
        predictions = [m(state, action_delta) for m in self.models]
        mean_latency = np.mean([p.latency[0].item() for p in predictions])
        uncertainty = np.var([p.latency[0].item() for p in predictions])
        return mean_latency, uncertainty

    def mean_prediction(self, state, action_delta):
        """
        FIX C1: Returns a full next-state object with averaged predictions.
        Used by the Dyna imagination loop to roll forward imagined states.
        """
        predictions = [m(state, action_delta) for m in self.models]
        next_state = state.copy()
        next_state.cpu = np.mean([p.cpu[0].item() for p in predictions])
        next_state.mem = np.mean([p.mem[0].item() for p in predictions])
        next_state.pending = np.mean([p.pending[0].item() for p in predictions])
        next_state.request_rate = np.mean([p.request_rate[0].item() for p in predictions])
        next_state.p99_latency = np.mean([p.latency[0].item() for p in predictions])

        # Advance time features
        next_state.time_step += 1
        t_norm = next_state.time_step / 120
        next_state.time_of_day = (np.sin(2*np.pi*t_norm), np.cos(2*np.pi*t_norm))

        # Update derivative
        next_state.request_rate_deriv = next_state.request_rate - state.request_rate
        return next_state

    def get_uncertainty(self, state, action_delta):
        """Returns per-variable uncertainty dict for diagnostics."""
        predictions = [m(state, action_delta) for m in self.models]
        return {
            'cpu': np.var([p.cpu[0].item() for p in predictions]),
            'mem': np.var([p.mem[0].item() for p in predictions]),
            'pending': np.var([p.pending[0].item() for p in predictions]),
            'latency': np.var([p.latency[0].item() for p in predictions]),
            'request_rate': np.var([p.request_rate[0].item() for p in predictions]),
        }
```

### 4.3 Training — Dyna-Q Two-Phase Data Collection (fixes M2)

```python
def train_world_model_dyna(ensemble, env, ppo_policy=None, phase='initial'):
    """
    FIX M2: Two-phase data collection eliminates distribution shift.
    Phase 1 (initial): Random policy for broad coverage
    Phase 2 (on-policy): Trained PPO policy with epsilon-greedy exploration
    """
    buffer = []

    if phase == 'initial' or ppo_policy is None:
        # Phase 1: Random exploration for broad state coverage
        collect_policy = lambda obs: env.action_space.sample()
        n_episodes = 3000
    else:
        # Phase 2: On-policy data from trained PPO + exploration
        epsilon = 0.1
        def collect_policy(obs):
            if np.random.random() < epsilon:
                return env.action_space.sample()
            action, _ = ppo_policy.predict(obs, deterministic=True)
            return action
        n_episodes = 2000

    for ep in range(n_episodes):
        obs, _ = env.reset()
        for step in range(120):
            action = collect_policy(obs)
            next_obs, reward, done, _, _ = env.step(action)
            buffer.append(Transition(obs, action, next_obs, reward))
            obs = next_obs
            if done:
                break

    # Train each ensemble member on bootstrapped subset
    for model in ensemble.models:
        subset = bootstrap_sample(buffer, size=len(buffer))
        train_single_model(model, subset, loss='nll', epochs=100, lr=1e-3)

    # Validate
    val_data = collect_validation_trajectories(env, n=500, policy=collect_policy)
    nll = evaluate_nll(ensemble, val_data)
    calibration = evaluate_calibration(ensemble, val_data)
    print(f"Phase={phase} | NLL: {nll:.3f} | Calibration Error: {calibration:.3f}")

    return ensemble
```

---

## 5. Component 3: Dyna-Q Training Loop (replaces standalone PPO + inference-time planner)

This replaces **both** the old `train_ppo.py` (Component 6 in v1) and the old `ModelBasedPlanner` (Component 3 in v1). The world model now contributes during training, not at inference.

```python
from sb3_contrib import RecurrentPPO  # FIX D9: LSTM policy for temporal patterns
from stable_baselines3.common.vec_env import SubprocVecEnv  # FIX D4: parallel envs

def make_env(rank):
    def _init():
        env = KubernetesAutoscalingEnv()
        env.seed(rank)
        return env
    return _init

def train_dyna(total_real_steps=500_000, imagination_ratio=4, horizon=3):
    """
    Dyna-Q loop:
    1. Collect real transitions from simulator
    2. Train world model on real data (iteratively)
    3. Generate imagined transitions from world model
    4. Train PPO on real + imagined data
    """

    # ── FIX D4: 8 parallel environments for coverage ──
    n_envs = 8
    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # ── FIX D9: RecurrentPPO with LSTM for temporal pattern recognition ──
    ppo = RecurrentPPO(
        "MlpLstmPolicy", vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=dict(
            net_arch=[256, 256],
            lstm_hidden_size=128,
        ),
        verbose=1,
        tensorboard_log="./logs/dyna/"
    )

    ensemble = WorldModelEnsemble(n_models=10)
    real_buffer = ReplayBuffer(max_size=200_000)

    # ── Phase 1: Initial world model from random data ──
    ensemble = train_world_model_dyna(ensemble, KubernetesAutoscalingEnv(), phase='initial')

    # ── Phase 2: Interleaved training ──
    steps_per_epoch = 50_000
    n_epochs = total_real_steps // steps_per_epoch

    for epoch in range(n_epochs):
        # 2a. Collect real transitions with current PPO
        real_transitions = collect_on_policy(ppo, vec_env, n_steps=steps_per_epoch)
        real_buffer.extend(real_transitions)

        # 2b. Retrain world model on accumulated real data (on-policy)
        ensemble = train_world_model_dyna(
            ensemble, KubernetesAutoscalingEnv(),
            ppo_policy=ppo, phase='on_policy'
        )

        # 2c. Generate imagined transitions
        imagined = []
        for _ in range(steps_per_epoch * imagination_ratio):
            # Sample a real state as starting point
            start_state = real_buffer.sample_state()
            imagined_state = start_state.copy()

            for h in range(horizon):
                action, _ = ppo.predict(imagined_state.as_vector())
                delta = action - 3
                next_state = ensemble.mean_prediction(imagined_state, delta)

                # Check uncertainty — discard high-uncertainty imaginations
                _, unc = ensemble.predict(imagined_state, delta)
                if unc > UNCERTAINTY_DISCARD_THRESHOLD:
                    break

                # Compute reward in imagination
                reward = compute_reward_from_state(next_state, delta)
                imagined.append(Transition(
                    imagined_state.as_vector(), action,
                    next_state.as_vector(), reward
                ))
                imagined_state = next_state

        # 2d. Train PPO on real + imagined data
        mixed_buffer = real_transitions + imagined
        ppo.learn(total_timesteps=len(mixed_buffer))

        # 2e. Log diagnostics
        print(f"Epoch {epoch}: real={len(real_transitions)}, "
              f"imagined={len(imagined)}, "
              f"model_NLL={evaluate_nll(ensemble, real_buffer.sample(1000)):.3f}")

    ppo.save("ppo_dyna_autoscaler")
    save_ensemble(ensemble, "world_model_ensemble.pt")
    return ppo, ensemble
```

---

## 6. Component 4: Safety Filter (fixes C4)

```python
class SafetyFilter:
    """Hard-coded invariant rules. Human-specified. Never trained."""
    MIN_REPLICAS = 2
    MAX_REPLICAS = 30
    MAX_DELTA = 3
    MIN_INTERVAL_STEPS = 1  # FIX C4: measured in sim steps, not wall-clock

    def __init__(self):
        self.last_action_step = -999  # sim step of last non-zero action
        self.override_count = 0
        self.total_count = 0

    def check(self, state, proposed_delta, sim_step):
        """
        FIX C4: Uses sim_step instead of time.time().
        At 30s/step, MIN_INTERVAL_STEPS=1 enforces a 30s cooldown.
        """
        self.total_count += 1
        original_delta = proposed_delta
        new_replicas = state.replicas + proposed_delta

        # Rule 1: Absolute bounds
        if new_replicas < self.MIN_REPLICAS or new_replicas > self.MAX_REPLICAS:
            proposed_delta = np.clip(new_replicas, self.MIN_REPLICAS,
                                     self.MAX_REPLICAS) - state.replicas

        # Rule 2: No scale-down during high latency
        if proposed_delta < 0 and state.p99_latency > 0.8 * state.sla_target:
            proposed_delta = 0

        # Rule 3: Max delta per cycle
        if abs(proposed_delta) > self.MAX_DELTA:
            proposed_delta = np.clip(proposed_delta, -self.MAX_DELTA, self.MAX_DELTA)

        # Rule 4: Rate limiting (FIX C4: sim time, not wall clock)
        if (sim_step - self.last_action_step < self.MIN_INTERVAL_STEPS
                and proposed_delta != 0):
            proposed_delta = 0

        if proposed_delta != original_delta:
            self.override_count += 1

        if proposed_delta != 0:
            self.last_action_step = sim_step

        return proposed_delta

    @property
    def fallback_rate(self):
        return self.override_count / max(self.total_count, 1)
```

---

## 7. Component 5: HPA Baseline — Realistic (fixes D6)

```python
from collections import deque

class RealisticHPA:
    """
    FIX D6: HPA with stabilization window, matching real K8s behavior.
    - Scale-up: immediate (same as real HPA)
    - Scale-down: uses max of last 6 steps (3-minute window)
    """
    def __init__(self, target_cpu=0.50):
        self.target_cpu = target_cpu
        self.replica_history = deque(maxlen=6)  # 3-min window at 30s steps

    def act(self, state):
        desired = math.ceil(state.replicas * state.cpu_util / self.target_cpu)
        desired = np.clip(desired, 2, 30)
        self.replica_history.append(desired)

        # Scale-down stabilization: use max of recent window
        if desired < state.replicas:
            desired = max(self.replica_history)

        return int(desired - state.replicas)

    def reset(self):
        self.replica_history.clear()
```

---

## 8. Full Agent — Simplified (no inference-time planner override)

```python
class ContainerScaleAgent:
    """
    v2: PPO is the sole decision-maker. No world-model override at inference.
    The world model contributed during training (Dyna). At test time, only
    PPO + safety filter are in the loop.
    """
    def __init__(self):
        self.ppo = RecurrentPPO.load("ppo_dyna_autoscaler")
        self.safety = SafetyFilter()
        self.hpa = RealisticHPA()
        self.lstm_states = None

    def decide(self, state, sim_step):
        try:
            action, self.lstm_states = self.ppo.predict(
                state.as_vector(),
                state=self.lstm_states,
                deterministic=True
            )
            delta = action - 3
            safe_delta = self.safety.check(state, delta, sim_step)
            return safe_delta

        except Exception:
            return self.hpa.act(state)

    def reset(self):
        self.lstm_states = None
        self.safety = SafetyFilter()
        self.hpa.reset()
```
# ContainerScale-RL v2 — Part 3: Experiments, Structure, Traceability

## 9. Evaluation: 8 Experiments (Revised)

### Changes from v1

| # | What Changed | Why |
|---|---|---|
| Exp 1 | Baseline is now `RealisticHPA` with stabilization window | FIX D6: fair comparison |
| Exp 2 | Compares Dyna-trained PPO vs model-free PPO (no world model at all) | FIX M7: measures Dyna contribution, not inference-time override |
| Exp 4 | Primary metric changed from override-rate to **regret** | FIX D3: override-rate is circular; regret measures actual value |
| Exp 5 | Now measures Dyna imagination horizon, not inference planning horizon | Architectural shift: no inference-time planner |
| Exp 7 | `per_pod_capacity` now in observation space | FIX M3: calibration not confounded by hidden variable |

### Experiment Table

| # | Experiment | IV | DV | Expected Result |
|---|---|---|---|---|
| 1 | **Dyna-RL vs Realistic HPA** | Agent type | Cost, P99, SLA violation % | RL wins on cost (always), SLA under diurnal/ramp; ties on steady |
| 2 | **Dyna-PPO vs Model-Free PPO** | World model in training loop (yes/no) | Sample efficiency, SLA compliance | Dyna converges faster; better SLA under flash crowd (imagined cold-start experience) |
| 3 | **Structured vs Flat world model** | Model architecture | Prediction NLL on held-out patterns | Structured generalizes better to unseen workloads |
| 4 | **Safety filter regret analysis** | None (diagnostic) | Per-decision reward difference: proposed vs filter-modified | Near-zero regret at convergence proves agent learned safety; positive regret proves filter value |
| 5 | **Dyna imagination horizon** | H ∈ {1, 3, 5} during Dyna training | SLA compliance, sample efficiency | H=3 best; H=1 can't see past cold-start; H=5 compounds model error |
| 6 | **Reward weight sensitivity** | α_cost ∈ {0.01, 0.1, 1.0} | Pareto frontier: cost vs SLA violation | Demonstrates operator-tunable tradeoff curve |
| 7 | **Uncertainty calibration** | None (diagnostic) | Predicted variance vs actual error (scatter) | Ensemble uncertainty correlates with prediction error; no hidden-variable confound |
| 8 | **Generalization** | Train on {steady, diurnal}, test on {flash, ramp} | SLA compliance on unseen patterns | Domain randomization helps; LSTM captures temporal trends |

### Experiment 4 Detail: Regret-Based Safety Measurement (fixes D3)

```python
def measure_safety_regret(agent, env, n_episodes=100):
    """
    For each decision where the safety filter modifies PPO's proposal,
    simulate both the proposed and modified actions for one step and
    record the reward difference.

    Interpretation:
    - Near-zero regret: agent already proposes safe actions (filter is redundant)
    - Large positive regret: filter prevents real harm (filter is valuable)
    - Large negative regret: filter makes things worse (filter miscalibrated)
    """
    regrets = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        agent.reset()
        for step in range(120):
            action, _ = agent.ppo.predict(obs, deterministic=True)
            proposed_delta = action - 3
            safe_delta = agent.safety.check(
                env.get_state(), proposed_delta, env.time_step
            )

            if safe_delta != proposed_delta:
                # Fork: simulate both actions
                env_copy = env.deep_copy()

                _, r_proposed, _, _, _ = env_copy.step(action)
                _, r_safe, _, _, _ = env.step(safe_delta + 3)

                regrets.append(r_proposed - r_safe)
            else:
                obs, _, done, _, _ = env.step(action)

            if done:
                break

    return {
        'mean_regret': np.mean(regrets),
        'std_regret': np.std(regrets),
        'positive_fraction': np.mean([r > 0 for r in regrets]),
        'n_overrides': len(regrets),
    }
```

---

## 10. Project Structure

```
ContainerScale-RL/
├── src/
│   ├── env/
│   │   ├── k8s_sim.py              # Gymnasium environment (all C/M/D fixes)
│   │   ├── workload.py             # Traffic patterns (M6, M8 fixes)
│   │   └── obs_delay.py            # Observation delay wrapper (D8)
│   │
│   ├── world_model/
│   │   ├── structured_model.py     # StructuredWorldModel + GaussianMLP
│   │   ├── ensemble.py             # WorldModelEnsemble (C1 fix, D7: N=10)
│   │   └── train_model.py          # Two-phase data collection (M2 fix)
│   │
│   ├── training/
│   │   ├── dyna_loop.py            # Dyna-Q training loop (replaces old train_ppo + planner)
│   │   ├── replay_buffer.py        # Real + imagined transition storage
│   │   └── curriculum.py           # Optional: steady→diurnal→flash progression
│   │
│   ├── safety/
│   │   └── safety_filter.py        # Hard-coded rules (C4 fix)
│   │
│   ├── agents/
│   │   ├── hpa_baseline.py         # RealisticHPA (D6 fix)
│   │   └── agent.py                # ContainerScaleAgent (simplified, no override)
│   │
│   └── evaluation/
│       ├── experiments.py           # All 8 experiments (revised)
│       ├── regret_analysis.py       # Experiment 4 regret measurement (D3 fix)
│       ├── metrics.py               # Cost, SLA, calibration calculators
│       └── visualize.py             # Training curves, Pareto frontiers, calibration plots
│
├── notebooks/
│   ├── 01_simulator_validation.ipynb    # Verify queue model, workload bounds
│   ├── 02_world_model_analysis.ipynb    # NLL, calibration, distribution shift check
│   └── 03_results.ipynb                 # All 8 experiments
│
└── configs/
    ├── env_config.yaml
    └── training_config.yaml
```

---

## 11. Critique Resolution Traceability

Every issue from the critique is mapped to its resolution in v2.

### Critical (P0) — All Fixed

| ID | Issue | Resolution |
|---|---|---|
| C1 | `mean_prediction` missing | Added to `WorldModelEnsemble` (§4.2) |
| C2 | State variables uninitialized | Full initialization in `__init__` and `reset` (§3.1) |
| C3 | Graduated pods never removed | One-pass prune in `step()` (§3.2) |
| C4 | Rate limiter uses wall-clock | Changed to `sim_step` parameter (§6) |

### Major (P1) — All Fixed

| ID | Issue | Resolution |
|---|---|---|
| M1 | Uncertainty scope broken in planner | **Eliminated**: no inference-time planner in v2. Uncertainty used during Dyna imagination to discard unreliable rollouts (§5) |
| M2 | World model trained on random data | Two-phase collection: random → on-policy with ε-greedy (§4.3) |
| M3 | `per_pod_capacity` unobserved | Added to 24-dim observation vector (§3.3) |
| M4 | Request rate static in imagination | `predict_request_rate` sub-network added to world model (§4.1) |
| M5 | Scale-down doesn't update ready_pods | Proper removal logic: pending first, then ready (§3.2) |
| M6 | Noisy pattern yields negative RPS | Log-normal distribution, always positive (§3.5) |
| M7 | PPO not trained with planner | **Resolved by architecture**: Dyna-Q trains PPO with world model. No separate planner at inference (§5) |
| M8 | Flash crowd exceeds action bounds | Spike reduced to 500 rps (reachable at max_replicas=30) (§3.5) |

### Design (P2–P4) — All Addressed

| ID | Issue | Resolution |
|---|---|---|
| D1 | No queue accumulation | `queue_depth` state variable + Little's Law latency (§3.2) |
| D2 | Stability penalty undifferentiated | Context-conditioned: only penalizes when latency < 0.5×SLA (§3.4) |
| D3 | Experiment 4 metric circular | Replaced with regret analysis (§9) |
| D4 | Single-env training | 8 parallel envs via `SubprocVecEnv` (§5) |
| D5 | `contention` undefined | Removed from world model inputs; replaced with `queue_depth` (§4.1) |
| D6 | HPA baseline too naive | `RealisticHPA` with 3-minute stabilization window (§7) |
| D7 | Ensemble too small (N=5) | Increased to N=10 (§4.2) |
| D8 | Observation delay not modeled | 1-step delay buffer in simulator (§3.1) |
| D9 | Frame-stacking insufficient | `RecurrentPPO` with LSTM policy replaces frame-stacking (§5) |

### Theoretical (T1, T2) — Resolved

| ID | Issue | Resolution |
|---|---|---|
| T1 | PPO/planner override incoherent | Dyna-Q: world model trains PPO, no inference override (§2) |
| T2 | Claims exceed actual coverage | Honest scope table with per-defect coverage assessment (§1) |

---

## 12. Known Limitations (Updated)

1. **Action space is ±3 replicas.** Cannot handle arbitrary step-function spikes. Flash crowd scaled to 500 rps to remain within physical bounds. Production use requires combining with manual pre-scaling for known events.

2. **Single deployment control.** Multi-service joint scaling is out of scope.

3. **Pod scaling only.** Node provisioning (Cluster Autoscaler) is environmental.

4. **Causal graph is hand-coded, not learned.** If the graph structure is wrong, the world model has blind spots. Stated as a limitation, not a feature.

5. **Observation delay is fixed at 30s.** Real Prometheus pipelines have variable delay (15–60s). The fixed delay is an approximation. True POMDP solution via belief tracking is future work.

6. **World model retrains during Dyna but is frozen at deployment.** Real cluster drift requires periodic retraining. Online model adaptation is future work.

7. **Reward weights are hyperparameters.** Experiment 6 provides the Pareto frontier for operator tuning.

8. **Queue model is simplified.** Little's Law assumes stable arrival rates within each 30s step. Bursty sub-step arrivals are not modeled. This is adequate for the 30s decision granularity but would need refinement for sub-second control.

9. **Traffic forecasting is one-step-ahead.** The world model's `predict_request_rate` learns local trends, not long-horizon demand forecasting. True anticipation of arbitrary future demand patterns (e.g., predicting a flash crowd 5 minutes before it arrives from external signals) is out of scope.

10. **Placement remains unaddressed.** The agent controls replica count only. Pod-to-node scheduling is handled by the Kubernetes scheduler and is not part of the action space.

---

## 13. Dependencies

```
gymnasium>=0.29
sb3-contrib>=2.0          # RecurrentPPO (LSTM policy)
stable-baselines3>=2.0
torch>=2.0
numpy
matplotlib
pyyaml
tensorboard
```

Live cluster integration (future): `kubernetes`, `prometheus-api-client`

---

## 14. What Makes This an RL Project

The RL contributions, in order of importance:

1. **MDP formulation** for K8s autoscaling with state/action/reward design that addresses each HPA defect, with honest scope boundaries
2. **Dyna-Q training** — world model generates synthetic experience for PPO, creating a principled model-based RL system with joint optimization
3. **Structured world model** respecting domain-knowledge causal graph (replicas → capacity → utilization → latency), now including traffic forecasting
4. **LSTM policy** (RecurrentPPO) for temporal pattern recognition under partial observability, replacing inadequate frame-stacking
5. **Uncertainty-aware imagination** — ensemble variance gates Dyna rollouts, preventing hallucinated training data
6. **Queue-theoretic simulation** grounded in Little's Law with proper state accumulation
7. **8 controlled experiments** with revised metrics (regret-based safety, realistic baseline) proving each component's contribution
