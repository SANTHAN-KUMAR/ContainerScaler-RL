# ContainerScale-RL

ContainerScale-RL is an autonomous Reinforcement Learning-based autoscaler for Kubernetes. It replaces the standard Horizontal Pod Autoscaler (HPA) by learning complex, non-linear traffic patterns (cold-start latency spikes, queue dynamics, diurnal patterns) using a two-phase simulation-to-real architecture.

The live deployment now targets **Google's Online Boutique** microservices demo running on a local k3s cluster, replacing the previous `podinfo` test app.

## Architecture Overview

The system operates in two distinct phases:

1.  **Phase 1: Simulation & Training (Offline)**
    *   **Simulator:** A custom Gymnasium environment (`K8sSimEnv`) that models Kubernetes queue dynamics, cold-start latency, and CPU/Memory utilization using Little's Law. Runs at 1000+ steps/second.
    *   **Agent:** A `RecurrentPPO` agent (from `sb3-contrib`) with LSTM memory to anticipate traffic patterns, plus a `QR-DQN` alternative with frame stacking. An `EnsembleMetaAgent` combines both.
    *   **Safety Filter:** A pure-Python layer enforcing hard invariants (min/max replicas, no scale-down during latency spikes).

2.  **Phase 2: Live Cluster Deployment (Online)**
    *   **Target App:** [Online Boutique](https://github.com/GoogleCloudPlatform/microservices-demo) — a realistic 11-service e-commerce app. The RL agent scales the `frontend` deployment.
    *   **Observer:** `PrometheusObserver` pulls live metrics via Prometheus (Traefik ingress metrics for HTTP traffic) and normalizes them into the exact 23-dimensional observation vector used during training.
    *   **Executor:** `K8sPatchExecutor` uses `kubectl scale` (bypasses Python TLS issues with local k3s) to apply replica changes every 5–30 seconds.
    *   **Fallback:** Any RL failure instantly falls back to `RealisticHPA` to guarantee SLA compliance.
    *   **Dashboard:** A Flask-based Live Command Center with real-time metrics, an embedded Online Boutique iframe (via reverse proxy), and one-click agent/traffic controls.

## Quickstart

### Local Setup
Requires Python 3.10+ and a running k3s cluster.

```bash
# Clone the repository
git clone https://github.com/SANTHAN-KUMAR/ContainerScaler-RL.git
cd ContainerScaler-RL

# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .[dev,loadtest]
```

### Phase 1: Train the RL Agent in Simulation

```bash
# Train the RecurrentPPO model (saves to ppo_autoscaler.zip)
crl-train --config configs/training_config.yaml

# Or train QR-DQN
python -m src.training.train_dqn --config configs/dqn_training_config.yaml
```

### Phase 2: Live Cluster Setup & Deployment

```bash
# Deploy Online Boutique + Prometheus to your k3s cluster
kubectl apply -f deploy/online-boutique.yaml
kubectl apply -f deploy/prometheus.yaml

# Run the live RL agent (defaults to EnsembleMetaAgent on the frontend deployment)
crl-live --prom http://localhost:30090 --namespace default --deployment frontend --model ensemble

# Or launch the full dashboard (recommended)
python -m src.dashboard.app
# Open http://localhost:5000
```

### Phase 3: Evaluate Performance

```bash
# Generate realistic e-commerce load (flash-crowd profile)
TRAFFIC_PROFILE=flash locust -f deploy/locustfile.py --headless -u 200 -r 30 --run-time 10m --host http://127.0.0.1:80

# Run simulation baseline experiment (RL vs HPA)
python -m src.evaluation.sim_experiments --exp 1

# Run generalization test on held-out patterns
python -m src.evaluation.sim_experiments --exp 9 --model ppo_autoscaler --episodes 50
```

## Directory Structure

*   `configs/`: YAML configurations for the simulator and training loops.
*   `deploy/`: Kubernetes manifests (`online-boutique.yaml`, `prometheus.yaml`), k3s networking fix, and Locust load-testing shapes.
*   `src/agents/`: RL agent inference wrapper, HPA baseline, and `EnsembleMetaAgent`.
*   `src/env/`: Gymnasium K8s Simulator and Synthetic Workload Generators.
*   `src/evaluation/`: Metric calculations, experiment runners, and plotting utilities.
*   `src/live/`: Prometheus observer (Traefik-aware), kubectl-based executor, and the live agent control loop.
*   `src/models/`: PyTorch World Models (Structured, Flat, Ensemble) for Research Experiment 3.
*   `src/safety/`: Hard-coded invariant rules for the safety filter.
*   `src/training/`: RecurrentPPO and QR-DQN training scripts with alpha-schedule callback.
*   `src/dashboard/`: Flask Live Command Center with real-time metrics and boutique proxy.

## Live Dashboard

The dashboard (`src/dashboard/app.py`) provides a full Live Command Center:

- **Real-time metrics** — RPS, P99 latency, CPU, replicas, agent decisions (reads directly from Prometheus when the agent is idle)
- **Online Boutique iframe** — embedded via Flask reverse proxy at `/boutique/`, auto-shows when the NodePort is live
- **Traffic profiles** — select `steady`, `flash`, `diurnal`, `ddos`, or `step` before injecting load
- **Model selector** — choose `ensemble`, `ppo_autoscaler`, `qr_dqn`, or `hpa` before deploying the agent
- **NodePort connectivity** — no port-forwards needed; Prometheus (`:30090`) and Boutique (`:30808`) are accessed directly

## Experiments

The project includes 9 experiments (8 sim, 1 live) to validate performance. See `src/evaluation/sim_experiments.py` for details.

| Exp | Description |
|-----|-------------|
| 1   | RL vs HPA across all 5 training patterns |
| 2–7 | Ablation studies (safety filter, reward shaping, LSTM, etc.) |
| 8   | Live sim-to-real gap analysis |
| 9   | Generalization on held-out patterns (`double_peak`, `sawtooth`) |