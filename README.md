# ContainerScale-RL

ContainerScale-RL is an autonomous Reinforcement Learning-based autoscaler for Kubernetes. It is designed to replace the standard Horizontal Pod Autoscaler (HPA) by learning complex, non-linear traffic patterns (such as cold-start latency spikes, queue dynamics, and diurnal patterns) using a two-phase simulation-to-real architecture.

## Architecture Overview

The system operates in two distinct phases:

1.  **Phase 1: Simulation & Training (Offline)**
    *   **Simulator:** A custom Gymnasium environment (`K8sSimEnv`) that models Kubernetes queue dynamics, cold-start latency, and CPU/Memory utilization using Little's Law. It runs at 1000+ steps/second.
    *   **Agent:** A `RecurrentPPO` agent (from `sb3-contrib`) equipped with LSTM memory to anticipate traffic patterns and handle partial observability.
    *   **Safety Filter:** A pure-Python layer that enforces hard invariants (e.g., minimum/maximum replicas, no scale-down during latency spikes).

2.  **Phase 2: Live Cluster Deployment (Online)**
    *   **Observer:** A bridge (`PrometheusObserver`) that pulls live metrics from the Kubernetes cluster (via Prometheus) and normalizes them into the exact 22-dimensional observation vector used during simulation.
    *   **Executor:** A Python script (`LiveClusterAgent`) that runs every 30 seconds, queries the RL agent for a scaling decision, passes it through the safety filter, and patches the Kubernetes Deployment API.
    *   **Fallback:** If the RL model fails or encounters a critical error, the agent instantly falls back to a deterministic `RealisticHPA` formula to guarantee SLA compliance.

## Quickstart

### Local Setup
Ensure you have Python 3.10+ installed.

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
# Train the RecurrentPPO model (will save to ppo_autoscaler.zip)
crl-train --config configs/training_config.yaml
```

### Phase 2: Live Cluster Setup & Deployment

We provide a lightweight setup for local testing using `k3s`.

```bash
# Setup local k3s cluster, deploy Prometheus, and the podinfo target app
chmod +x deploy/k3s-setup.sh
sudo ./deploy/k3s-setup.sh

# Run the live RL agent
crl-live --prom http://localhost:30090 --namespace default --deployment podinfo --model ppo_autoscaler
```

### Phase 3: Evaluate Performance

```bash
# Generate synthetic load using Locust to test the agent's scaling behavior
locust -f deploy/locustfile.py --headless -u 100 -r 10 --run-time 1h --host http://<podinfo-svc-ip>:9898

# Run simulation baseline experiment (RL vs HPA)
python -m src.evaluation.sim_experiments --exp 1
```

## Directory Structure

*   `configs/`: YAML configurations for the simulator and training loops.
*   `deploy/`: Kubernetes manifests (podinfo, prometheus), k3s setup, and Locust load testing shapes.
*   `src/agents/`: RL Agent inference wrapper and Realistic HPA baseline.
*   `src/env/`: Gymnasium K8s Simulator and Synthetic Workload Generators.
*   `src/evaluation/`: Metric calculations, experiment runners, and plotting utilities.
*   `src/live/`: Prometheus observer, Kubernetes executor, and the live agent control loop.
*   `src/models/`: PyTorch World Models (Structured, Flat, Ensemble) for Research Experiment 3.
*   `src/safety/`: Hard-coded invariant rules for the safety filter.
*   `src/training/`: RecurrentPPO orchestrator script.

## Experiments
The project includes a suite of 8 experiments (7 sim, 1 live) to validate performance. See `src/evaluation/sim_experiments.py` for implementation details.