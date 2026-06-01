# Implementation Status Report: ContainerScale-RL

This document tracks the current state of the repository against the original `PROJECT_PROPOSAL.md` and documents all recent changes.

**Last updated:** June 2, 2026

---

## Executive Summary

**Status: 100% Core Completion + Active Research Improvements**

Every component from the original proposal is fully implemented. Recent work has focused on three areas: (1) upgrading the observation space and reward function for better training stability, (2) adding a second RL algorithm (QR-DQN) as an alternative to RecurrentPPO, and (3) expanding the evaluation suite with generalization testing on held-out workload patterns.

---

## Recent Changes (Latest Commits)

### Live Target: podinfo → Online Boutique
- **Files changed:** `src/live/observer.py`, `src/live/live_agent.py`, `src/live/executor.py`, `deploy/locustfile.py`, `src/dashboard/app.py`
- Migrated the live deployment target from the simple `podinfo` Go app to **Google's Online Boutique** (`frontend` deployment)
- Online Boutique is a realistic 11-service e-commerce demo — provides meaningful HTTP traffic patterns (browse, cart, checkout)
- HTTP metrics now sourced from **Traefik ingress** (`traefik_service_requests_total`, `traefik_service_request_duration_seconds_bucket`) since the boutique frontend doesn't natively expose Prometheus metrics
- CPU normalisation updated: per-pod limit changed from `250m` → `100m` (boutique frontend resource profile)
- Memory normalisation updated: per-pod limit changed from `512Mi` → `128Mi`
- Default deployment arg changed from `podinfo` → `frontend` across all entry points

### Executor: Python K8s API → kubectl
- **Files changed:** `src/live/executor.py`
- Replaced `apps_v1.patch_namespaced_deployment_scale()` with `subprocess.check_call(["kubectl", "scale", ...])` to bypass Python TLS authentication issues with local k3s self-signed certificates
- Same approach applied in `observer.py` for fetching replica counts

### EnsembleMetaAgent as Default Live Controller
- **Files changed:** `src/live/live_agent.py`
- `--model` default changed from `ppo_autoscaler` → `ensemble`
- `live_agent.py` now explicitly routes `model="ensemble"` to load `EnsembleMetaAgent` and `model="hpa"` to force HPA-only mode (bypasses RL entirely)
- Added `--interval` CLI argument (default `30`s) to control the step interval independently of `--steps`

### Dashboard: NodePort-First Architecture
- **Files changed:** `src/dashboard/app.py`, `src/dashboard/static/app.js`, `src/dashboard/static/live.js`, `src/dashboard/templates/live.html`
- Removed port-forward processes (`port_forward_prom`, `port_forward_podinfo`) — replaced with direct NodePort access
  - Prometheus: `http://127.0.0.1:30090`
  - Online Boutique: `http://127.0.0.1:30808`
- Added `_check_nodeport()` helper for TCP-level connectivity checks (treats timeout as "alive" to avoid false alarms under DDOS load)
- **Live metrics without agent:** When the RL agent is not running, `/api/live_metrics` now fetches metrics directly from Prometheus via a `PrometheusObserver` instance — the dashboard stays "Live" permanently
- **Boutique reverse proxy:** New `/boutique/<path>` route proxies the Online Boutique through Flask (same origin), bypassing browser cross-origin iframe restrictions. Rewrites `href`, `src`, and `action` attributes to keep navigation within the proxy.
- **Boutique iframe panel:** Live Command Center now embeds the boutique storefront in a side panel. Auto-shows when NodePort is detected live; user can toggle visibility.
- **Configurable start parameters:** `toggleService()` in `app.js` now sends a JSON payload when starting services — `traffic_profile` for Locust, `model` for the live agent. UI dropdowns drive these values.
- Live agent now started with `--steps 720 --interval 5` (was `--steps 120` with no interval flag)
- Locust now targets `http://127.0.0.1:80` (Traefik ingress) with `200` users / `30` spawn rate (was `9898` port, `100` users)

### Locust: Realistic E-Commerce User Journeys
- **Files changed:** `deploy/locustfile.py`
- Completely rewrote from a single `GET /delay/1` task to a full `BoutiqueShopperUser` with weighted tasks:
  - `view_homepage` (weight 5), `view_product` (weight 2), `add_to_cart_and_view` (weight 2), `checkout` (weight 1)
  - Think-time: `between(0.5, 2.0)` seconds
- Added 4 traffic shape profiles selectable via `TRAFFIC_PROFILE` env var:
  - `flash` — ramp 20→200 users, sustain, cool-down (mirrors `K8sSimEnv` flash_crowd scenario)
  - `diurnal` — 1-hour sine wave (125 ± 75 users)
  - `ddos` — instant spike to 500 users
  - `step` — 50→100→150→200 users in 60s increments
  - `steady` (default) — constant load via CLI `--users` flag

### Prometheus: Enhanced Pod Scrape Config
- **Files changed:** `deploy/prometheus.yaml`
- Added relabelling rules to the pod scrape config:
  - Honour `prometheus.io/port` annotation for per-pod port override
  - Honour `prometheus.io/path` annotation for custom metrics paths
  - Preserve `app` and `pod` labels for PromQL filtering

### Observer: TLS + Traefik Fixes
- **Files changed:** `src/live/observer.py`
- Disabled SSL verification (`verify_ssl = False`) and suppressed `urllib3` InsecureRequestWarning for local k3s self-signed certs
- PromQL queries updated to use Traefik ingress metrics instead of app-level `http_requests_total`
- Replica count now fetched via `kubectl get deploy` subprocess (same TLS bypass as executor)
- Container name matching updated: looks for `"server"` or the deployment name (boutique frontend container is named `"server"`)
- Added `Gi` memory unit parsing

---

### Observation Space: 22 → 23 Dimensions
- **Files changed:** `src/env/k8s_sim.py`, `src/live/observer.py`, `src/safety/safety_filter.py`
- Added **traffic acceleration** (2nd derivative of request rate) as `obs[22]`, normalized by `/500.0`
- Tells the agent whether traffic is speeding up or slowing down — improves anticipation of flash crowds and ramp-downs
- `SafetyFilter.from_obs()` is forward-compatible: only reads indices 0–8 and 20, unaffected by the expansion
- `PrometheusObserver` updated to compute and include the acceleration feature from live Prometheus data

### Reward Function: Normalized Multi-Objective
- **Files changed:** `src/env/k8s_sim.py`, `configs/env_config.yaml`
- **Old:** Additive `r_sla + r_cost + r_crash + r_stability` with fixed weights (`-10.0`, `-0.01`, `-50.0`, `-0.1`) — SLA and cost were on incompatible scales
- **New:** Both SLA and cost normalized to `[-1, 0]` independently, then combined via `alpha`:
  ```
  reward = alpha × r_cost_normalized + (1 - alpha) × r_sla_normalized
  ```
  - `alpha = 0.3` default (30% cost focus, 70% SLA focus)
  - Crash is now a **multiplicative amplifier** (`× 3.0`) rather than an additive `-50` penalty — avoids scale explosion
  - Stability penalty reduced to `-0.02` per `|delta|` to avoid discouraging emergency scaling
- Eliminates the 1000:1 penalty asymmetry that caused the agent to ignore cost entirely

### Alpha Schedule (Cyclical Annealing)
- **Files changed:** `src/training/train_ppo.py`, `src/training/train_dqn.py`
- **New file:** `src/training/alpha_schedule.py`
- `AlphaScheduleCallback` dynamically adjusts `alpha` during training:
  ```
  Steps 0–28%:    alpha = 0.1  (SLA-focused — learn to keep latency low first)
  Steps 28–71%:   alpha → 0.4  (anneal toward cost awareness)
  Steps 71–100%:  alpha = 0.2  (settle at balanced tradeoff)
  ```
- Prevents the agent from learning cost-cutting before it has mastered SLA compliance

### Multi-Algorithm Inference Agent
- **Files changed:** `src/agents/agent.py`
- `ContainerScaleAgent` now auto-detects and loads any of three model types:
  1. **QR-DQN** (tried first — preferred for new training runs)
  2. **RecurrentPPO** (fallback — loads existing trained models)
  3. **DQN** (final fallback)
- Added **frame stacking support** for DQN/QR-DQN: maintains a rolling buffer of the last N observations to give temporal context without LSTM
- `reset()` now clears both `lstm_states` and `obs_buffer`

### QR-DQN Training Pipeline
- **Files changed:** `src/training/train_dqn.py`, `configs/dqn_training_config.yaml`
- QR-DQN (Quantile Regression DQN) added as an alternative to RecurrentPPO
- Uses frame stacking (4 frames × 23 dims = 92-dim input) for temporal context
- `total_timesteps` increased to 700,000 (from 500,000)
- `norm_reward=False` — reward is pre-normalized by the new reward function, so VecNormalize reward scaling is disabled

### Training Config Updates
- **Files changed:** `configs/training_config.yaml`, `configs/dqn_training_config.yaml`
- PPO: `total_timesteps` 500k → 700k, `n_envs` 20 → 8 (better RAM/GPU balance)
- DQN: `total_timesteps` 500k → 700k

### Held-Out Workload Patterns
- **Files changed:** `src/env/workload.py`
- Added 2 new patterns **never used during training** — for generalization testing only:
  - `double_peak`: Two sine peaks at t=0.25 and t=0.75 (~350 rps each) — tests multi-modal anticipation
  - `sawtooth`: Linear ramp 50→300 rps, instant drop, repeat 3× — tests rapid recovery
- `HELD_OUT_PATTERNS` and `ALL_PATTERNS` constants exported for evaluation scripts

### Experiment 9: Generalization Test
- **Files changed:** `src/evaluation/sim_experiments.py`
- New `run_exp9_generalization()` function tests the trained agent on held-out patterns
- Reports a **generalization gap** — if gap > 10%, overfitting is flagged
- `composite_score()` helper combines SLA compliance and cost efficiency into a single comparable number
- Experiment 1 now includes per-pattern breakdown in addition to overall averages
- CLI updated: `--model` and `--episodes` flags added; `--exp 9` routes to generalization test

---

## Component Status

### 1. Simulator (Gymnasium Environment)
**Status: Fully Implemented — Recently Enhanced**
- **Files:** `src/env/k8s_sim.py`, `src/env/workload.py`
- 23-dimensional MDP state vector (was 22 — added traffic acceleration)
- Little's Law queue dynamics and cold-start delays fully modeled
- Normalized multi-objective reward with alpha-based cost-SLA tradeoff
- `set_alpha()` method for dynamic reward adjustment during training
- 5 training workload patterns + 2 held-out evaluation patterns

### 2. RL Agent (RecurrentPPO + QR-DQN)
**Status: Fully Implemented — Multi-Algorithm**
- **Files:** `src/training/train_ppo.py`, `src/training/train_dqn.py`, `src/training/alpha_schedule.py`, `src/agents/agent.py`
- RecurrentPPO (LSTM) trained for 700k+ timesteps — checkpoints in `checkpoints/`
- QR-DQN with frame stacking as alternative algorithm
- Alpha schedule callback for cyclical cost-SLA annealing during training
- `ContainerScaleAgent` auto-detects model type (QR-DQN / RecurrentPPO / DQN)
- LSTM state and frame buffer both managed correctly across steps

### 3. Safety Filter
**Status: Fully Implemented — Forward-Compatible**
- **Files:** `src/safety/safety_filter.py`
- 4 hard-coded invariants: replica bounds, no scale-down during high latency, max delta ±3, rate limiting
- `from_obs()` updated to be forward-compatible with observation space expansions (only reads indices 0–8 and 20)

### 4. HPA Baseline
**Status: Fully Implemented**
- **Files:** `src/agents/hpa_baseline.py`
- Faithful Kubernetes HPA formula with 3-minute scale-down stabilization window

### 5. World Model (Experiment 3 Only)
**Status: Fully Implemented**
- **Files:** `src/models/structured_model.py`, `src/models/flat_model.py`, `src/models/ensemble.py`
- 5-part probabilistic ensemble (CPU, Mem, Pending, RPS, Latency)
- Not part of the agent's decision path

### 6. Phase 2: Live Cluster Deployment
**Status: Fully Implemented — Migrated to Online Boutique**
- **Infrastructure:** `deploy/k3s-setup.sh`, `deploy/online-boutique.yaml`, `deploy/prometheus.yaml`, `deploy/locustfile.py`
- **Live Agent:** `src/live/observer.py`, `src/live/executor.py`, `src/live/live_agent.py`, `src/live/metrics_logger.py`
- Target deployment migrated from `podinfo` → Online Boutique `frontend`
- `PrometheusObserver` updated: Traefik ingress metrics, kubectl-based replica fetch, TLS bypass for k3s
- `K8sPatchExecutor` updated: uses `kubectl scale` subprocess instead of Python K8s API (TLS fix)
- `LiveClusterAgent` updated: explicit `ensemble`/`hpa`/`<path>` routing, configurable `--interval` flag
- Control loop: configurable interval (default 5s for dashboard, 30s for CLI), Prometheus → EnsembleMetaAgent → SafetyFilter → kubectl

### 7. Evaluation Pipeline
**Status: Fully Implemented — Expanded**
- **Files:** `src/evaluation/sim_experiments.py`, `src/evaluation/live_experiment.py`, `src/evaluation/metrics.py`, `src/evaluation/visualize.py`, `src/evaluation/pygame_visualizer.py`
- Experiments 1–8 from proposal: fully functional
- **Experiment 9 (new):** Generalization test on held-out workload patterns
- Composite score metric for single-number RL vs HPA comparison
- Per-pattern breakdown in Experiment 1
- Pygame-based real-time visualizer added

---

## Bonus Feature: Live Management Dashboard
**Status: Fully Implemented — Major Overhaul (Exceeds Proposal)**
- **Files:** `src/dashboard/app.py`, `src/dashboard/templates/`, `src/dashboard/static/`
- NodePort-first architecture — no port-forward processes required
- Live metrics fetched directly from Prometheus when agent is idle (always-on dashboard)
- Embedded Online Boutique storefront via Flask reverse proxy (`/boutique/`)
- Configurable traffic profiles (steady/flash/diurnal/ddos/step) and model selector (ensemble/ppo/qr-dqn/hpa)
- Real-time pod grid, SLA breach indicator, agent decision log

---

## Known Limitations (Unchanged from Proposal)

1. **±3 replica action space** — cannot handle arbitrary step-function spikes
2. **Single deployment** — multi-service joint scaling out of scope
3. **Pod scaling only** — node provisioning handled by Cluster Autoscaler
4. **Sim-to-real gap** — domain randomization mitigates but doesn't eliminate
5. **Local k3s ≠ production cloud** — network and scheduling differ
6. **LSTM anticipation is implicit** — not an explicit traffic forecaster

---

## Quick Reference: Run Commands

```bash
# Train RecurrentPPO (700k steps, 8 envs)
crl-train --config configs/training_config.yaml

# Train QR-DQN (700k steps, 4 envs)
python -m src.training.train_dqn --config configs/dqn_training_config.yaml

# Run Experiment 1 (RL vs HPA, all patterns)
python -m src.evaluation.sim_experiments --exp 1 --model ppo_autoscaler --episodes 100

# Run Experiment 9 (Generalization test on held-out patterns)
python -m src.evaluation.sim_experiments --exp 9 --model ppo_autoscaler --episodes 50

# Run Experiment 8 (Live/Sim RL vs HPA comparison)
python -m src.evaluation.live_experiment --mode sim --workload diurnal

# Deploy live agent (EnsembleMetaAgent on Online Boutique frontend)
crl-live --prom http://localhost:30090 --namespace default --deployment frontend --model ensemble

# Deploy live agent (HPA-only mode)
crl-live --prom http://localhost:30090 --namespace default --deployment frontend --model hpa

# Launch the full dashboard
python -m src.dashboard.app
# Open http://localhost:5000

# Inject flash-crowd traffic via Locust
TRAFFIC_PROFILE=flash locust -f deploy/locustfile.py --headless -u 200 -r 30 --run-time 10m --host http://127.0.0.1:80
```
