# Component: Training Script (train_ppo.py)

## Responsibility

Wires together the Simulator and RecurrentPPO to run the training loop. Initializes the agent, runs 500,000 training steps across 8 parallel environments, logs progress, and saves the trained model weights to disk.

This component contains no learning logic itself — it is purely configuration and orchestration. All learning happens inside SB3's RecurrentPPO implementation.

---

## Inputs and Outputs

| | Description |
|---|---|
| **Input** | Training configuration (hyperparameters from `training_config.yaml`) |
| **Input** | K8sSimEnv (8 parallel instances) |
| **Output** | Saved model weights file (`ppo_autoscaler.zip`) |
| **Output** | TensorBoard logs (`./logs/ppo/`) |

---

## Responsibilities

1. Load training configuration from config file
2. Spin up 8 parallel simulator environments using `SubprocVecEnv`
3. Initialize `RecurrentPPO` with correct architecture and hyperparameters
4. Run `model.learn(total_timesteps=500_000)`
5. Write TensorBoard logs during training
6. Save trained model weights to disk on completion

---

## Hyperparameters

| Parameter | Value | What It Controls |
|---|---|---|
| `n_envs` | 8 | Number of parallel simulators running simultaneously |
| `total_timesteps` | 500,000 | Total training steps across all environments |
| `learning_rate` | 3e-4 | How fast the agent updates its policy |
| `n_steps` | 2048 | Steps collected per environment before each update |
| `batch_size` | 64 | Mini-batch size for gradient updates |
| `n_epochs` | 10 | How many passes over collected data per update |
| `gamma` | 0.99 | Discount factor — how much future rewards matter |
| `net_arch` | [256, 256] | Two hidden layers of 256 neurons each |
| `lstm_hidden_size` | 128 | LSTM memory capacity |

---

## Why 8 Parallel Environments

Each environment runs a different randomly-selected workload pattern and cluster configuration. Running 8 in parallel means every training batch contains diverse experience — the agent sees steady traffic, flash crowds, diurnal patterns, and ramps all within the same update cycle.

Without parallelism, the agent might see 2048 steps of the same pattern before updating, which leads to overfitting to that pattern.

---

## Training Loop (What SB3 Does Internally)

The training script calls `model.learn()` which runs this loop internally:

```
repeat until 500,000 steps reached:
    for each of 8 environments:
        collect 2048 steps of (state, action, reward, next_state)
    
    total batch = 8 × 2048 = 16,384 transitions
    
    for 10 epochs:
        sample mini-batches of 64 from the batch
        compute PPO loss (policy + value + entropy)
        run gradient update
    
    log metrics to TensorBoard
    clear batch, repeat
```

The training script does not implement this loop — it just calls `model.learn(500_000)` and SB3 handles everything above.

---

## TensorBoard Logs

Metrics written during training for monitoring:

| Metric | What It Shows |
|---|---|
| `rollout/ep_rew_mean` | Average episode reward — primary training signal |
| `rollout/ep_len_mean` | Average episode length (should be 120) |
| `train/policy_loss` | Policy network loss |
| `train/value_loss` | Value function loss |
| `train/entropy_loss` | Exploration entropy |
| `train/approx_kl` | How much policy changed per update |

A healthy training run shows `ep_rew_mean` steadily increasing and then plateauing.

---

## Model Saving

On completion, the trained model is saved as a single `.zip` file:

```
ppo_autoscaler.zip
    ├── policy weights (neural network parameters)
    ├── LSTM weights
    ├── value function weights
    └── training metadata
```

This file is the **only output that Phase 2 depends on**. It is the bridge between training and live deployment.

---

## Failure Modes

| Failure | What Happens | Mitigation |
|---|---|---|
| Training diverges | Reward collapses or oscillates wildly | Monitor TensorBoard; reduce learning rate |
| Out of memory | 8 parallel envs exceed RAM | Reduce `n_envs` to 4 |
| Training too slow | 500k steps takes too long | Verify simulator runs at 1000+ steps/sec |
| Model not saved | Crash before completion | Add checkpoint saving every 50k steps |

---

## Dependencies

| Dependency | Purpose |
|---|---|
| `sb3_contrib.RecurrentPPO` | The RL algorithm |
| `stable_baselines3.common.vec_env.SubprocVecEnv` | Parallel environment runner |
| `K8sSimEnv` | The simulator being trained on |
| `tensorboard` | Training metrics visualization |
| `pyyaml` | Loading config from `training_config.yaml` |

---

## Configuration File (training_config.yaml)

All hyperparameters are externalized to a config file so they can be changed without modifying code:

```yaml
training:
  total_timesteps: 500000
  n_envs: 8
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  lstm_hidden_size: 128
  net_arch: [256, 256]
  save_path: "ppo_autoscaler"
  log_dir: "./logs/ppo/"
```
