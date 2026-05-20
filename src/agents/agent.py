"""
ContainerScaleAgent — RL inference wrapper with HPA fallback.

Loads a trained RecurrentPPO model, maintains LSTM hidden state across
steps, and falls back to RealisticHPA on any inference failure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.hpa_baseline import RealisticHPA
from src.safety.safety_filter import ClusterState, SafetyFilter

logger = logging.getLogger(__name__)


class ContainerScaleAgent:
    """RL autoscaling agent with safety filter and HPA fallback.

    Parameters
    ----------
    model_path : str | Path
        Path to the saved RecurrentPPO model (without .zip extension).
    sla_target : float
        P99 latency SLA target in ms.
    """

    def __init__(
        self,
        model_path: str | Path = "ppo_autoscaler",
        sla_target: float = 200.0,
    ) -> None:
        # Lazy import to avoid hard dependency at module level
        from sb3_contrib import RecurrentPPO

        self.model_path = Path(model_path)
        self.sla_target = sla_target

        # Load the trained model
        logger.info("Loading RecurrentPPO model from %s", self.model_path)
        self.ppo = RecurrentPPO.load(str(self.model_path))
        logger.info("Model loaded successfully")

        # LSTM hidden state — persists across calls for temporal memory
        # LSTM hidden state — persists across calls for temporal memory
        self.lstm_states: Any = None

        # Safety filter + fallback
        self.safety = SafetyFilter(sla_target=sla_target)
        self.hpa = RealisticHPA()

        # Check for VecNormalize statistics file
        self.vec_normalize = None
        vec_norm_path = self.model_path.parent / f"{self.model_path.stem}_vecnormalize.pkl"
        if vec_norm_path.exists():
            from stable_baselines3.common.vec_env import VecNormalize
            from stable_baselines3.common.vec_env import DummyVecEnv
            from src.env.k8s_sim import K8sSimEnv

            logger.info("Loading VecNormalize stats from %s", vec_norm_path)
            try:
                # Create a dummy environment to load VecNormalize stats
                dummy_venv = DummyVecEnv([lambda: K8sSimEnv(config_path=None)])
                self.vec_normalize = VecNormalize.load(str(vec_norm_path), dummy_venv)
                self.vec_normalize.training = False  # Freeze statistics updates
                logger.info("VecNormalize stats loaded successfully")
            except Exception as e:
                logger.error("Failed to load VecNormalize stats, proceeding without normalization: %s", e)

    def decide(self, obs: np.ndarray, step: int) -> int:
        """Produce a safe replica delta from the current observation.

        Parameters
        ----------
        obs : np.ndarray
            22-dimensional observation vector (normalized).
        step : int
            Current step number (for safety filter cooldown).

        Returns
        -------
        int
            Safe replica delta in [-3, +3].
        """
        try:
            # Normalize observation for the PPO network if normalizer is available
            network_obs = obs
            if self.vec_normalize is not None:
                obs_batch = np.expand_dims(obs, axis=0)
                network_obs = self.vec_normalize.normalize_obs(obs_batch)[0]

            # Run RL inference with LSTM state
            action, self.lstm_states = self.ppo.predict(
                network_obs,
                state=self.lstm_states,
                deterministic=True,
            )
            # Map action [0..6] → delta [-3..+3]
            delta = int(action) - 3

            # Build state snapshot for safety filter from raw observation
            state = ClusterState.from_obs(obs, sla_target=self.sla_target)

            # Apply safety rules
            return self.safety.check(state, delta, step)

        except Exception:
            logger.exception("RL inference failed — falling back to HPA")
            state = ClusterState.from_obs(obs, sla_target=self.sla_target)
            return self.hpa.act(state)

    def decide_with_info(self, obs: np.ndarray, step: int) -> dict[str, Any]:
        """Like decide(), but returns detailed info for logging/analysis.

        Returns
        -------
        dict
            Keys: ``delta``, ``proposed_delta``, ``source`` ("rl" | "hpa").
        """
        try:
            # Normalize observation for the PPO network if normalizer is available
            network_obs = obs
            if self.vec_normalize is not None:
                obs_batch = np.expand_dims(obs, axis=0)
                network_obs = self.vec_normalize.normalize_obs(obs_batch)[0]

            action, self.lstm_states = self.ppo.predict(
                network_obs,
                state=self.lstm_states,
                deterministic=True,
            )
            proposed_delta = int(action) - 3
            state = ClusterState.from_obs(obs, sla_target=self.sla_target)
            safe_delta = self.safety.check(state, proposed_delta, step)

            return {
                "delta": safe_delta,
                "proposed_delta": proposed_delta,
                "source": "rl",
            }

        except Exception:
            logger.exception("RL inference failed — falling back to HPA")
            state = ClusterState.from_obs(obs, sla_target=self.sla_target)
            hpa_delta = self.hpa.act(state)

            return {
                "delta": hpa_delta,
                "proposed_delta": hpa_delta,
                "source": "hpa",
            }

    def reset(self) -> None:
        """Reset LSTM state, safety filter, and HPA for a new run."""
        self.lstm_states = None
        self.safety.reset()
        self.hpa.reset()

    def __repr__(self) -> str:
        return f"ContainerScaleAgent(model={self.model_path})"
