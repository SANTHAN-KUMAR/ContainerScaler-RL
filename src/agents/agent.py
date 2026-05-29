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
        Path to the saved model (without .zip extension).
    sla_target : float
        P99 latency SLA target in ms.
    """

    def __init__(
        self,
        model_path: str | Path = "ppo_autoscaler",
        sla_target: float = 200.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.sla_target = sla_target

        # Load the trained model dynamically based on algorithm type
        self.model = None
        self.model_type = None

        # Try loading as QRDQN
        try:
            from sb3_contrib import QRDQN
            self.model = QRDQN.load(str(self.model_path))
            self.model_type = "qrdqn"
            logger.info("Loaded QRDQN model from %s", self.model_path)
        except Exception:
            pass

        # Try loading as RecurrentPPO if QRDQN failed
        if self.model is None:
            try:
                from sb3_contrib import RecurrentPPO
                self.model = RecurrentPPO.load(str(self.model_path))
                self.model_type = "recurrent_ppo"
                logger.info("Loaded RecurrentPPO model from %s", self.model_path)
            except Exception:
                pass

        # Try loading as DQN if both QRDQN and RecurrentPPO failed
        if self.model is None:
            try:
                from stable_baselines3 import DQN
                self.model = DQN.load(str(self.model_path))
                self.model_type = "dqn"
                logger.info("Loaded DQN model from %s", self.model_path)
            except Exception:
                pass

        if self.model is None:
            raise ValueError(
                f"Could not load model from {self.model_path}. "
                "Ensure it is a valid QRDQN, RecurrentPPO, or DQN zip file."
            )

        # LSTM hidden state — persists across calls for temporal memory (RecurrentPPO only)
        self.lstm_states: Any = None

        # Frame stacking buffer (DQN/QRDQN only, initialized on first step)
        self.obs_buffer: np.ndarray | None = None

        # Safety filter + fallback
        self.safety = SafetyFilter(sla_target=sla_target)
        self.hpa = RealisticHPA()

        # Check for VecNormalize statistics file
        self.vec_normalize = None
        vec_norm_path = self.model_path.parent / f"{self.model_path.stem}_vecnormalize.pkl"
        if vec_norm_path.exists():
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecFrameStack
            from src.env.k8s_sim import K8sSimEnv

            logger.info("Loading VecNormalize stats from %s", vec_norm_path)
            try:
                # Create a dummy environment to load VecNormalize stats
                dummy_venv = DummyVecEnv([lambda: K8sSimEnv(config_path=None)])
                
                # If the model uses frame stacking, wrap the dummy environment too
                obs_dim = 23
                model_dim = self.model.observation_space.shape[0]
                if model_dim > obs_dim:
                    dummy_venv = VecFrameStack(dummy_venv, n_stack=model_dim // obs_dim)

                self.vec_normalize = VecNormalize.load(str(vec_norm_path), dummy_venv)
                self.vec_normalize.training = False  # Freeze statistics updates
                logger.info("VecNormalize stats loaded successfully")
            except Exception as e:
                logger.error("Failed to load VecNormalize stats, proceeding without normalization: %s", e)

    def _predict_proposed_delta(self, obs: np.ndarray) -> int:
        """Helper to run the neural network prediction on the observation (with framing/norm)."""
        obs_dim = obs.shape[0]
        model_dim = self.model.observation_space.shape[0]

        # Handle frame stacking if the model expects more dimensions than a single observation
        if model_dim > obs_dim:
            n_stack = model_dim // obs_dim
            if self.obs_buffer is None:
                self.obs_buffer = np.zeros(model_dim, dtype=np.float32)
                self.obs_buffer[-obs_dim:] = obs
            else:
                self.obs_buffer = np.roll(self.obs_buffer, -obs_dim)
                self.obs_buffer[-obs_dim:] = obs
            network_obs = self.obs_buffer
        else:
            network_obs = obs

        # Normalize observation if a normalizer is available
        if self.vec_normalize is not None:
            obs_batch = np.expand_dims(network_obs, axis=0)
            network_obs = self.vec_normalize.normalize_obs(obs_batch)[0]

        # Run RL inference
        if self.model_type == "recurrent_ppo":
            action, self.lstm_states = self.model.predict(
                network_obs,
                state=self.lstm_states,
                deterministic=True,
            )
        else:
            action, _ = self.model.predict(
                network_obs,
                deterministic=True,
            )

        # Map action [0..6] → delta [-3..+3]
        return int(action) - 3

    def decide(self, obs: np.ndarray, step: int) -> int:
        """Produce a safe replica delta from the current observation.

        Parameters
        ----------
        obs : np.ndarray
            23-dimensional observation vector (normalized).
        step : int
            Current step number (for safety filter cooldown).

        Returns
        -------
        int
            Safe replica delta in [-3, +3].
        """
        try:
            proposed_delta = self._predict_proposed_delta(obs)

            # Build state snapshot for safety filter from raw observation
            state = ClusterState.from_obs(obs, sla_target=self.sla_target)

            # Apply safety rules
            return self.safety.check(state, proposed_delta, step)

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
            proposed_delta = self._predict_proposed_delta(obs)
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
        """Reset LSTM state, frame buffer, safety filter, and HPA for a new run."""
        self.lstm_states = None
        self.obs_buffer = None
        self.safety.reset()
        self.hpa.reset()

    def __repr__(self) -> str:
        return f"ContainerScaleAgent(model={self.model_path}, type={self.model_type})"
