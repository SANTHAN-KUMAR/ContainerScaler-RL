"""
Training Script — DQN / QRDQN for K8sSimEnv.

Off-policy, value-based alternative to RecurrentPPO.
Uses frame stacking (VecFrameStack) for temporal context instead of LSTM.

Key differences from PPO:
  - Off-policy: uses a replay buffer → much more sample efficient
  - Value-based: learns Q(s,a) directly instead of a policy π(a|s)
  - Higher GPU utilization: continuous gradient updates from replay buffer
  - QRDQN variant: models the full return distribution → risk-aware decisions

Usage:
    python -m src.training.train_dqn
    python -m src.training.train_dqn --algo dqn --device cuda
    python -m src.training.train_dqn --config configs/dqn_training_config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize

from src.env.k8s_sim import K8sSimEnv

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "dqn_training_config.yaml"


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
        Path to dqn_training_config.yaml.

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
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                vec_env.save(str(path))
        return res


def build_model(algo: str, env, cfg: dict):
    """Instantiate the requested algorithm with config parameters.

    Parameters
    ----------
    algo : str
        Algorithm name: ``"dqn"`` or ``"qrdqn"``.
    env : VecEnv
        Vectorized environment (already wrapped with VecFrameStack + VecNormalize).
    cfg : dict
        Training configuration dict.

    Returns
    -------
    BaseAlgorithm
        The constructed SB3 model, ready for ``.learn()``.
    """
    # Base policy kwargs — net_arch controls the MLP width
    policy_kwargs = dict(
        net_arch=cfg["net_arch"],
    )

    # Common constructor kwargs shared between DQN and QRDQN
    common_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg["learning_rate"],
        buffer_size=cfg["buffer_size"],
        learning_starts=cfg["learning_starts"],
        batch_size=cfg["batch_size"],
        gamma=cfg["gamma"],
        tau=cfg.get("tau", 0.005),
        train_freq=cfg.get("train_freq", 4),
        gradient_steps=cfg.get("gradient_steps", 2),
        exploration_fraction=cfg.get("exploration_fraction", 0.2),
        exploration_initial_eps=cfg.get("exploration_initial_eps", 1.0),
        exploration_final_eps=cfg.get("exploration_final_eps", 0.02),
        tensorboard_log=cfg["log_dir"],
        device=cfg.get("device", "auto"),
        seed=cfg.get("seed", 42),
        verbose=1,
    )

    if algo == "dqn":
        from stable_baselines3 import DQN

        common_kwargs["policy_kwargs"] = policy_kwargs
        model = DQN(
            **common_kwargs,
            target_update_interval=cfg.get("target_update_interval", 1000),
        )
        logger.info("Created DQN model (net_arch=%s)", cfg["net_arch"])

    elif algo == "qrdqn":
        from sb3_contrib import QRDQN

        # QRDQN accepts n_quantiles via policy_kwargs
        n_quantiles = cfg.get("n_quantiles", 200)
        policy_kwargs["n_quantiles"] = n_quantiles
        common_kwargs["policy_kwargs"] = policy_kwargs

        model = QRDQN(**common_kwargs)
        logger.info(
            "Created QRDQN model (n_quantiles=%d, net_arch=%s)",
            n_quantiles, cfg["net_arch"],
        )

    else:
        raise ValueError(f"Unknown algorithm: {algo!r}. Choose 'dqn' or 'qrdqn'.")

    return model


def train(config_path: str | Path | None = None, algo_override: str | None = None) -> Path:
    """Run the full DQN/QRDQN training loop.

    Parameters
    ----------
    config_path : str | Path | None
        Path to dqn_training_config.yaml.  ``None`` uses default.
    algo_override : str | None
        Override the algorithm specified in config (``"dqn"`` or ``"qrdqn"``).

    Returns
    -------
    Path
        Path to the saved model file (without .zip extension).
    """
    config_path = Path(config_path) if config_path else _DEFAULT_CONFIG
    cfg = load_config(config_path)

    algo = algo_override or cfg.get("algorithm", "qrdqn")
    n_envs = cfg["n_envs"]
    seed = cfg.get("seed", 42)
    save_path = cfg["save_path"]
    frame_stack = cfg.get("frame_stack", 4)
    gamma = cfg["gamma"]

    logger.info(
        "Config: algo=%s n_envs=%d frame_stack=%d buffer=%d device=%s",
        algo, n_envs, frame_stack, cfg["buffer_size"], cfg.get("device", "auto"),
    )

    # ── Parallel environments ────────────────────────────────────────
    logger.info("Creating %d parallel environments (seed=%d)", n_envs, seed)
    raw_env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])

    # ── Frame stacking — gives temporal context without LSTM ─────────
    # Concatenates last N observations: 22-dim × 4 frames = 88-dim input
    stacked_env = VecFrameStack(raw_env, n_stack=frame_stack)
    logger.info(
        "VecFrameStack enabled: %d frames → obs shape %s",
        frame_stack, stacked_env.observation_space.shape,
    )

    # ── VecNormalize — reward normalization ──────────────────────────
    env = VecNormalize(
        stacked_env,
        norm_obs=False,       # obs already normalized in the environment
        norm_reward=True,     # normalize rewards with running variance
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
    )
    logger.info("VecNormalize enabled — reward normalization active")

    # ── Build model ──────────────────────────────────────────────────
    # Override log/save paths based on actual algorithm
    cfg["log_dir"] = f"./logs/{algo}/"
    if algo_override and algo_override != cfg.get("algorithm"):
        cfg["save_path"] = f"{algo}_autoscaler"
        save_path = cfg["save_path"]

    model = build_model(algo, env, cfg)

    # Log parameter count for GPU utilization context
    total_params = sum(p.numel() for p in model.policy.parameters())
    logger.info("Total trainable parameters: %s", f"{total_params:,}")

    # ── Checkpoint callback ──────────────────────────────────────────
    checkpoint_freq = cfg.get("checkpoint_freq", 50000)
    checkpoint_cb = CheckpointWithNormalizerCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path="./checkpoints/",
        name_prefix=f"{algo}_autoscaler",
        verbose=1,
    )

    # ── Train ────────────────────────────────────────────────────────
    total_timesteps = cfg["total_timesteps"]
    logger.info("Starting %s training for %d timesteps", algo.upper(), total_timesteps)

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    # ── Save model + normalizer stats ────────────────────────────────
    model_path = Path(save_path)
    model.save(str(model_path))
    logger.info("Model saved to %s.zip", model_path)

    vec_norm_path = model_path.parent / f"{model_path.stem}_vecnormalize.pkl"
    env.save(str(vec_norm_path))
    logger.info("VecNormalize stats saved to %s", vec_norm_path)
    logger.info(
        "NOTE: Load both files for inference — model: %s.zip, normalizer: %s",
        model_path, vec_norm_path,
    )

    env.close()
    return model_path


def main() -> None:
    """CLI entry point for DQN/QRDQN training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train ContainerScale-RL agent with DQN or QRDQN"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to dqn_training_config.yaml",
    )
    parser.add_argument(
        "--algo", type=str, default=None, choices=["dqn", "qrdqn"],
        help="Override algorithm (default: read from config, usually qrdqn)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (e.g. 'cuda', 'cpu'). Default: read from config.",
    )
    args = parser.parse_args()

    # Allow --device CLI flag to override config
    if args.device:
        config_path = Path(args.config) if args.config else _DEFAULT_CONFIG
        cfg = load_config(config_path)
        cfg["device"] = args.device
        # Write back temporarily — simpler than plumbing through
        import tempfile
        import os
        full_cfg = {"training": cfg}
        tmp_path = Path(tempfile.mktemp(suffix=".yaml", dir="."))
        try:
            with open(tmp_path, "w") as f:
                yaml.dump(full_cfg, f, default_flow_style=False)
            train(config_path=tmp_path, algo_override=args.algo)
        finally:
            if tmp_path.exists():
                os.unlink(tmp_path)
    else:
        train(config_path=args.config, algo_override=args.algo)


if __name__ == "__main__":
    main()
