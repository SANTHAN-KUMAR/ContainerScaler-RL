"""
Training Script — Orchestrates RecurrentPPO training on K8sSimEnv.

This is pure configuration and orchestration. All learning happens
inside SB3's RecurrentPPO implementation.

Usage:
    python -m src.training.train_ppo
    python -m src.training.train_ppo --config configs/training_config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from src.env.k8s_sim import K8sSimEnv

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "training_config.yaml"


def make_env(rank: int, seed: int = 0):
    """Factory function for creating a seeded K8sSimEnv instance.

    Parameters
    ----------
    rank : int
        Environment index (0 to n_envs-1).
    seed : int
        Base random seed.

    Returns
    -------
    callable
        A zero-argument function that returns a Gymnasium environment.
    """
    def _init():
        env = K8sSimEnv(seed=seed + rank, workload_pattern="random")
        return env
    return _init


def load_config(path: str | Path) -> dict:
    """Load training configuration from YAML.

    Parameters
    ----------
    path : str | Path
        Path to training_config.yaml.

    Returns
    -------
    dict
        The ``training`` section of the config file.
    """
    with open(path) as f:
        return yaml.safe_load(f)["training"]


class CheckpointWithNormalizerCallback(CheckpointCallback):
    """Custom checkpoint callback that also saves VecNormalize statistics."""
    def _on_step(self) -> bool:
        res = super()._on_step()
        if res and self.n_calls % self.save_freq == 0:
            path = Path(self.save_path) / f"{self.name_prefix}_{self.num_timesteps}_steps_vecnormalize.pkl"
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(str(path))
        return res


def train(config_path: str | Path | None = None) -> Path:
    """Run the full RecurrentPPO training loop.

    Parameters
    ----------
    config_path : str | Path | None
        Path to training_config.yaml.  ``None`` uses default.

    Returns
    -------
    Path
        Path to the saved model file (without .zip extension).
        A companion VecNormalize stats file is saved alongside it.
    """
    # Lazy import — only needed when actually training
    from sb3_contrib import RecurrentPPO

    config_path = Path(config_path) if config_path else _DEFAULT_CONFIG
    cfg = load_config(config_path)

    n_envs = cfg["n_envs"]
    seed = cfg.get("seed", 42)
    save_path = cfg["save_path"]
    log_dir = cfg["log_dir"]
    n_steps = cfg["n_steps"]
    batch_size = cfg["batch_size"]
    n_epochs = cfg["n_epochs"]
    gamma = cfg["gamma"]
    gae_lambda = cfg.get("gae_lambda", 0.95)
    ent_coef = cfg.get("ent_coef", 0.0)
    vf_coef = cfg.get("vf_coef", 0.5)
    max_grad_norm = cfg.get("max_grad_norm", 0.5)
    clip_range = cfg.get("clip_range", 0.2)

    logger.info(
        "Loaded config: %s (n_envs=%d n_steps=%d batch=%d epochs=%d)",
        config_path,
        n_envs,
        n_steps,
        batch_size,
        n_epochs,
    )

    logger.info("Creating %d parallel environments (seed=%d)", n_envs, seed)

    # ── Parallel environments ────────────────────────────────────────
    raw_env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])

    # ── VecNormalize — fixes the critic by keeping rewards in [-1, 1] ──
    # norm_obs=False : observations are already normalized in the environment
    # norm_reward=True: normalizes rewards using a running estimate of variance
    # clip_obs=10.0  : clips normalized obs to [-10, 10] to prevent outliers
    # clip_reward=10.0: clips normalized rewards to [-10, 10]
    # gamma          : must match PPO gamma for correct return normalization
    env = VecNormalize(
        raw_env,
        norm_obs=False,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
    )
    logger.info("VecNormalize enabled — reward normalization is active, observation normalization is disabled")

    # ── RecurrentPPO ─────────────────────────────────────────────────
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=cfg["learning_rate"],
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        clip_range=clip_range,
        policy_kwargs=dict(
            net_arch=cfg["net_arch"],
            lstm_hidden_size=cfg["lstm_hidden_size"],
        ),
        tensorboard_log=log_dir,
        device=cfg.get("device", "auto"),
        seed=seed,
        verbose=1,
    )

    # ── Checkpoint callback ──────────────────────────────────────────
    checkpoint_freq = cfg.get("checkpoint_freq", 50000)
    checkpoint_cb = CheckpointWithNormalizerCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path="./checkpoints/",
        name_prefix="ppo_autoscaler",
        verbose=1,
    )

    # ── Train ────────────────────────────────────────────────────────
    total_timesteps = cfg["total_timesteps"]
    logger.info("Starting training for %d timesteps", total_timesteps)

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    # ── Save model + normalizer stats ────────────────────────────────
    # The VecNormalize running stats (mean/variance) must be saved alongside
    # the model so inference uses the same normalization as training.
    model_path = Path(save_path)
    model.save(str(model_path))
    logger.info("Model saved to %s.zip", model_path)

    vec_norm_path = model_path.parent / f"{model_path.stem}_vecnormalize.pkl"
    env.save(str(vec_norm_path))
    logger.info("VecNormalize stats saved to %s", vec_norm_path)
    logger.info(
        "NOTE: Load both files for inference — "
        "model: %s.zip, normalizer: %s", model_path, vec_norm_path
    )

    env.close()
    return model_path


def main() -> None:
    """CLI entry point for training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train ContainerScale-RL agent")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to training_config.yaml",
    )
    args = parser.parse_args()

    train(config_path=args.config)


if __name__ == "__main__":
    main()
