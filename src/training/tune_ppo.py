"""
Hyperparameter Tuning Script — Optuna-based search for RecurrentPPO.

Runs N short trials (tuning_timesteps each) to find the best hyperparameter
combination, then saves the best config ready for a full training run.

Usage:
    python -m src.training.tune_ppo
    python -m src.training.tune_ppo --trials 30 --timesteps 50000
    python -m src.training.tune_ppo --trials 20 --timesteps 100000 --n-envs 20
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "training_config.yaml"
_BEST_PARAMS_PATH = Path("configs/best_hyperparams.json")
_TUNED_CONFIG_PATH = Path("configs/training_config_tuned.yaml")


# ── Hyperparameter search space ──────────────────────────────────────────────

def sample_hyperparams(trial) -> dict:
    """Define the search space for Optuna to explore.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object used to sample values.

    Returns
    -------
    dict
        Sampled hyperparameter dictionary.
    """
    # Learning rate — log-uniform is standard for LR search
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    # Batch size — must be a power of 2 for GPU efficiency
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    # Steps per rollout — how much data to collect before each update
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])

    # PPO epochs — how many gradient passes over each batch
    n_epochs = trial.suggest_int("n_epochs", 5, 20)

    # Discount factor — higher = more long-term thinking
    gamma = trial.suggest_float("gamma", 0.95, 0.999)

    # LSTM hidden size — memory capacity of the recurrent policy
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])

    # Network width — MLP layers before/after LSTM
    net_arch_size = trial.suggest_categorical("net_arch_size", [128, 256, 512])

    # GAE lambda — bias/variance tradeoff for advantage estimation
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)

    # Entropy coefficient — encourages exploration
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.01, log=True)

    # Value function coefficient
    vf_coef = trial.suggest_float("vf_coef", 0.3, 0.9)

    # Max gradient norm — prevents exploding gradients
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)

    # PPO clip range — limits policy update size
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "lstm_hidden_size": lstm_hidden_size,
        "net_arch": [net_arch_size, net_arch_size],
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "clip_range": clip_range,
    }


# ── Objective function ────────────────────────────────────────────────────────

def make_objective(n_envs: int, tuning_timesteps: int, seed: int, device: str):
    """Create the Optuna objective function.

    Parameters
    ----------
    n_envs : int
        Number of parallel environments per trial.
    tuning_timesteps : int
        Timesteps per trial (short budget for fast evaluation).
    seed : int
        Base random seed.
    device : str
        PyTorch device string ("auto", "cuda", "cpu").

    Returns
    -------
    callable
        Objective function that takes an Optuna trial and returns a score.
    """
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from src.env.k8s_sim import K8sSimEnv

    def objective(trial) -> float:
        params = sample_hyperparams(trial)

        logger.info(
            "Trial %d | lr=%.2e batch=%d n_steps=%d lstm=%d",
            trial.number,
            params["learning_rate"],
            params["batch_size"],
            params["n_steps"],
            params["lstm_hidden_size"],
        )

        # Build envs
        def make_env(rank: int):
            def _init():
                return K8sSimEnv(seed=seed + rank, workload_pattern="random")
            return _init

        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=params["gamma"],
        )

        try:
            model = RecurrentPPO(
                "MlpLstmPolicy",
                env,
                learning_rate=params["learning_rate"],
                n_steps=params["n_steps"],
                batch_size=params["batch_size"],
                n_epochs=params["n_epochs"],
                gamma=params["gamma"],
                gae_lambda=params["gae_lambda"],
                ent_coef=params["ent_coef"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                clip_range=params["clip_range"],
                policy_kwargs=dict(
                    net_arch=params["net_arch"],
                    lstm_hidden_size=params["lstm_hidden_size"],
                ),
                device=device,
                seed=seed,
                verbose=0,  # suppress per-step logs during tuning
            )

            model.learn(total_timesteps=tuning_timesteps, progress_bar=False)

            # ── Evaluate: run 10 episodes and average the episode reward ──
            env.training = False
            env.norm_reward = False
            episode_rewards = []
            obs = env.reset()
            lstm_states = None
            episode_starts = np.ones((n_envs,), dtype=bool)
            episode_reward = np.zeros(n_envs)
            steps = 0
            max_eval_steps = 10 * 120  # 10 episodes × 120 steps each

            while steps < max_eval_steps:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                obs, rewards, dones, _ = env.step(action)
                episode_reward += rewards
                episode_starts = dones

                for i, done in enumerate(dones):
                    if done:
                        episode_rewards.append(episode_reward[i])
                        episode_reward[i] = 0.0

                steps += 1

            mean_reward = float(np.mean(episode_rewards)) if episode_rewards else float("-inf")
            logger.info("Trial %d → mean_reward=%.2f", trial.number, mean_reward)
            return mean_reward

        except Exception as e:
            logger.warning("Trial %d failed: %s", trial.number, e)
            return float("-inf")

        finally:
            env.close()

    return objective


# ── Save results ──────────────────────────────────────────────────────────────

def save_best_params(best_params: dict, base_config_path: Path) -> None:
    """Save best hyperparams as JSON and as a ready-to-use training config YAML.

    Parameters
    ----------
    best_params : dict
        Best hyperparameters found by Optuna.
    base_config_path : Path
        Path to the base training_config.yaml to merge with.
    """
    # Save raw JSON
    _BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info("Best hyperparams saved to %s", _BEST_PARAMS_PATH)

    # Merge into a full training config YAML
    with open(base_config_path) as f:
        base_cfg = yaml.safe_load(f)

    tuned_cfg = base_cfg.copy()
    tuned_cfg["training"].update({
        "learning_rate": best_params["learning_rate"],
        "batch_size": best_params["batch_size"],
        "n_steps": best_params["n_steps"],
        "n_epochs": best_params["n_epochs"],
        "gamma": best_params["gamma"],
        "lstm_hidden_size": best_params["lstm_hidden_size"],
        "net_arch": best_params["net_arch"],
        "gae_lambda": best_params["gae_lambda"],
        "ent_coef": best_params["ent_coef"],
        "vf_coef": best_params["vf_coef"],
        "max_grad_norm": best_params["max_grad_norm"],
        "clip_range": best_params["clip_range"],
    })

    with open(_TUNED_CONFIG_PATH, "w") as f:
        yaml.dump(tuned_cfg, f, default_flow_style=False, sort_keys=False)
    logger.info("Tuned training config saved to %s", _TUNED_CONFIG_PATH)
    logger.info("Run full training with: crl-train --config %s", _TUNED_CONFIG_PATH)


# ── Main ──────────────────────────────────────────────────────────────────────

def tune(
    n_trials: int = 20,
    tuning_timesteps: int = 50_000,
    n_envs: int = 20,
    seed: int = 42,
    device: str = "auto",
    config_path: Path | None = None,
) -> dict:
    """Run Optuna hyperparameter search.

    Parameters
    ----------
    n_trials : int
        Number of trials to run.
    tuning_timesteps : int
        Timesteps per trial (short budget).
    n_envs : int
        Parallel environments per trial.
    seed : int
        Random seed.
    device : str
        PyTorch device.
    config_path : Path | None
        Base config path for merging results.

    Returns
    -------
    dict
        Best hyperparameters found.
    """
    import optuna
    from optuna.samplers import TPESampler

    # Suppress Optuna's verbose internal logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(
        "Starting Optuna search: %d trials × %d timesteps each",
        n_trials,
        tuning_timesteps,
    )
    logger.info("Each trial uses %d parallel envs on device=%s", n_envs, device)
    logger.info(
        "Estimated time: ~%d min (assuming ~2 min/trial)",
        n_trials * 2,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        study_name="containerscale_rl_ppo",
    )

    objective = make_objective(
        n_envs=n_envs,
        tuning_timesteps=tuning_timesteps,
        seed=seed,
        device=device,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["net_arch"] = [best.pop("net_arch_size")] * 2  # reconstruct list form

    logger.info("=" * 60)
    logger.info("Best trial: #%d  score=%.4f", study.best_trial.number, study.best_value)
    for k, v in best.items():
        logger.info("  %-25s %s", k, v)
    logger.info("=" * 60)

    save_best_params(best, config_path or _DEFAULT_CONFIG)
    return best


def main() -> None:
    """CLI entry point for hyperparameter tuning."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Tune ContainerScale-RL hyperparameters with Optuna")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials (default: 20)")
    parser.add_argument(
        "--timesteps", type=int, default=50_000,
        help="Timesteps per trial — shorter = faster but noisier (default: 50000)",
    )
    parser.add_argument("--n-envs", type=int, default=20, help="Parallel envs per trial (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="auto", help="PyTorch device (default: auto)")
    parser.add_argument("--config", type=str, default=None, help="Base training_config.yaml path")
    args = parser.parse_args()

    tune(
        n_trials=args.trials,
        tuning_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
        config_path=Path(args.config) if args.config else None,
    )


if __name__ == "__main__":
    main()
