"""
EnsembleMetaAgent — Mixture-of-Experts autoscaler.

Routes each scaling decision to the best specialist RL model based on
real-time workload classification from a sliding window of observations.

Specialist routing (calibrated from 4,900-episode benchmark):
  ┌──────────────────┬────────────────┬──────────────────────────────┐
  │ Classified As    │ Routed To      │ Why                          │
  ├──────────────────┼────────────────┼──────────────────────────────┤
  │ noisy            │ QRDQN          │ Distributional RL handles    │
  │                  │                │   stochastic loads           │
  │ ramp / sawtooth  │ DQN            │ Best trend-following on      │
  │                  │                │   linear increases           │
  │ steady / spike   │ PPO 1M         │ Most cost-efficient on       │
  │                  │                │   stable & spike patterns    │
  │ uncertain        │ PPO 700k       │ Safest all-rounder, never    │
  │                  │                │   collapses on any pattern   │
  └──────────────────┴────────────────┴──────────────────────────────┘

Usage:
    from src.agents.ensemble_agent import EnsembleMetaAgent
    agent = EnsembleMetaAgent()
    delta = agent.decide(obs, step)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.agent import ContainerScaleAgent
from src.safety.safety_filter import ClusterState, SafetyFilter

logger = logging.getLogger(__name__)


# ======================================================================
# Workload Classifier
# ======================================================================

@dataclass
class WorkloadFeatures:
    """Statistical features extracted from a sliding window of request rates."""
    cv: float = 0.0              # Coefficient of variation (σ/μ)
    slope: float = 0.0           # Linear trend slope (rps per step)
    delta_variance: float = 0.0  # Variance of rate-of-change
    spike_ratio: float = 0.0     # Max(|Δrate|) / mean(rate)
    mean_rate: float = 0.0       # Average request rate in window


@dataclass
class ClassifierThresholds:
    """Decision boundaries calibrated from empirical workload statistics.

    These values are derived from analyzing 50 seeds × 12 windows per
    pattern across all 7 workload types. They sit at the natural decision
    boundaries between pattern clusters.

    Calibration data (mean CV per pattern):
      noisy=0.632, double_peak=0.239, sawtooth=0.140, steady=0.091,
      flash_crowd=0.085, diurnal=0.064, gradual_ramp=0.062

    Calibration data (mean slope per pattern):
      sawtooth=6.50, gradual_ramp=3.35, double_peak=1.94,
      flash_crowd=0.63, noisy=0.39, diurnal=-0.34, steady=-0.02
    """
    # CV > this → classify as 'noisy'
    # Separates noisy (0.632) from everything else (max 0.239)
    noisy_cv: float = 0.35

    # |slope| > this → classify as 'ramp' or 'sawtooth'
    # Separates ramp (3.35) and sawtooth (6.50) from the rest (max 1.94)
    ramp_slope: float = 2.5

    # spike_ratio > this → classify as 'spike'
    # Noisy is excluded first (has high spike_ratio too)
    spike_ratio: float = 0.20

    # Slope > this with high delta variance → 'sawtooth' vs 'ramp'
    # Sawtooth has slope ≈ 6.5 vs ramp ≈ 3.35
    sawtooth_slope: float = 5.0


class WorkloadClassifier:
    """Real-time workload pattern classifier using sliding window statistics.

    Analyzes the last ``window_size`` observations to classify the current
    workload into one of: noisy, ramp, sawtooth, spike, steady.

    The classifier uses a priority-based rule chain:
      1. noisy:    high coefficient of variation
      2. ramp:     strong positive linear trend (moderate slope)
      3. sawtooth: very strong positive trend (steep slope)
      4. spike:    large sudden rate changes
      5. steady:   fallback (low CV, no trend, no spikes)

    Parameters
    ----------
    window_size : int
        Number of recent observations to consider. Default 10 (= 5 minutes
        at 30s/step). This balances reaction speed with classification
        stability. Configurable via constructor.
    thresholds : ClassifierThresholds | None
        Custom decision boundaries. None uses the data-calibrated defaults.
    """

    def __init__(
        self,
        window_size: int = 10,
        thresholds: ClassifierThresholds | None = None,
    ) -> None:
        self.window_size = window_size
        self.thresholds = thresholds or ClassifierThresholds()
        self._rate_history: deque[float] = deque(maxlen=window_size)
        self._classification: str = "uncertain"
        self._features: WorkloadFeatures = WorkloadFeatures()
        self._confidence: float = 0.0

    def update(self, obs: np.ndarray) -> str:
        """Update the classifier with a new observation.

        Parameters
        ----------
        obs : np.ndarray
            23-dimensional observation vector from K8sSimEnv.

        Returns
        -------
        str
            Classified pattern: 'noisy', 'ramp', 'sawtooth', 'spike',
            'steady', or 'uncertain' (insufficient data).
        """
        # Extract request rate from observation (obs[4] = rate / 500)
        request_rate = obs[4] * 500.0
        self._rate_history.append(request_rate)

        if len(self._rate_history) < self.window_size:
            self._classification = "uncertain"
            self._confidence = 0.0
            return self._classification

        rates = np.array(self._rate_history)
        self._features = self._compute_features(rates)
        self._classification, self._confidence = self._classify(self._features)
        return self._classification

    def _compute_features(self, rates: np.ndarray) -> WorkloadFeatures:
        """Extract statistical features from a window of request rates."""
        mean = np.mean(rates)
        std = np.std(rates)
        cv = std / max(mean, 1e-6)

        # Linear regression slope
        x = np.arange(len(rates), dtype=np.float64)
        slope = float(np.polyfit(x, rates, 1)[0])

        # Rate-of-change statistics
        deltas = np.diff(rates)
        delta_var = float(np.var(deltas))
        spike_ratio = float(np.max(np.abs(deltas)) / max(mean, 1e-6))

        return WorkloadFeatures(
            cv=cv, slope=slope, delta_variance=delta_var,
            spike_ratio=spike_ratio, mean_rate=mean,
        )

    def _classify(self, f: WorkloadFeatures) -> tuple[str, float]:
        """Apply the priority rule chain to classify the pattern."""
        t = self.thresholds

        # Rule 1: Noisy — high coefficient of variation
        if f.cv > t.noisy_cv:
            confidence = min(1.0, f.cv / (t.noisy_cv * 2))
            return "noisy", confidence

        # Rule 2: Sawtooth — very steep positive slope
        if f.slope > t.sawtooth_slope:
            confidence = min(1.0, f.slope / (t.sawtooth_slope * 2))
            return "sawtooth", confidence

        # Rule 3: Ramp — moderate positive slope
        if f.slope > t.ramp_slope:
            confidence = min(1.0, f.slope / (t.ramp_slope * 2))
            return "ramp", confidence

        # Rule 4: Spike — large sudden rate changes (but not noisy)
        if f.spike_ratio > t.spike_ratio:
            confidence = min(1.0, f.spike_ratio / (t.spike_ratio * 3))
            return "spike", confidence

        # Rule 5: Steady — default fallback
        confidence = 1.0 - f.cv  # higher CV = less confident it's steady
        return "steady", max(0.0, confidence)

    @property
    def classification(self) -> str:
        """Current workload classification."""
        return self._classification

    @property
    def features(self) -> WorkloadFeatures:
        """Features from the most recent window."""
        return self._features

    @property
    def confidence(self) -> float:
        """Confidence in the current classification [0, 1]."""
        return self._confidence

    def reset(self) -> None:
        """Clear history for a new episode."""
        self._rate_history.clear()
        self._classification = "uncertain"
        self._features = WorkloadFeatures()
        self._confidence = 0.0


# ======================================================================
# Routing Table
# ======================================================================

# Maps classified pattern → specialist name
ROUTING_TABLE: dict[str, str] = {
    "noisy": "qrdqn",
    "ramp": "dqn",
    "sawtooth": "dqn",
    "spike": "ppo_1m",
    "steady": "ppo_1m",
    "uncertain": "ppo",  # fallback: safest all-rounder
}


# ======================================================================
# Ensemble Meta-Agent
# ======================================================================

class EnsembleMetaAgent:
    """Mixture-of-Experts agent that routes to specialist RL models.

    Maintains a ``WorkloadClassifier`` that classifies the current traffic
    pattern from a sliding window of observations, then routes each
    ``decide()`` call to the best specialist model for that pattern.

    Parameters
    ----------
    model_paths : dict[str, str] | None
        Map of specialist name → model file path. None uses defaults.
    window_size : int
        Classifier sliding window size (steps). Default 10.
    thresholds : ClassifierThresholds | None
        Custom classifier thresholds. None uses data-calibrated defaults.
    sla_target : float
        P99 latency SLA target in ms.
    """

    def __init__(
        self,
        model_paths: dict[str, str] | None = None,
        window_size: int = 10,
        thresholds: ClassifierThresholds | None = None,
        sla_target: float = 200.0,
    ) -> None:
        self.sla_target = sla_target

        # Default model paths
        if model_paths is None:
            model_paths = {
                "ppo": "ppo_autoscaler",
                "ppo_1m": "ppo_autoscaler_1M_steps",
                "dqn": "dqn_autoscaler",
                "qrdqn": "qrdqn_autoscaler",
            }

        # Load specialist agents
        self.specialists: dict[str, ContainerScaleAgent] = {}
        self._fallback_name: str = "ppo"

        for name, path in model_paths.items():
            try:
                self.specialists[name] = ContainerScaleAgent(
                    model_path=path, sla_target=sla_target,
                )
                logger.info("Loaded specialist: %s from %s", name, path)
            except Exception as e:
                logger.warning("Could not load specialist %s: %s", name, e)

        if not self.specialists:
            raise ValueError("No specialist models could be loaded!")

        # Set fallback to first available specialist
        if self._fallback_name not in self.specialists:
            self._fallback_name = next(iter(self.specialists))
            logger.warning(
                "Primary fallback 'ppo' not available, using '%s'",
                self._fallback_name,
            )

        # Classifier
        self.classifier = WorkloadClassifier(
            window_size=window_size, thresholds=thresholds,
        )

        # Routing stats (for analysis)
        self._routing_counts: dict[str, int] = {}
        self._routing_log: list[dict[str, Any]] = []

    def decide(self, obs: np.ndarray, step: int) -> int:
        """Produce a safe replica delta using the best specialist.

        Parameters
        ----------
        obs : np.ndarray
            23-dimensional observation vector.
        step : int
            Current step number.

        Returns
        -------
        int
            Safe replica delta in [-3, +3].
        """
        # Update classifier
        pattern = self.classifier.update(obs)

        # Route to specialist
        specialist_name = ROUTING_TABLE.get(pattern, self._fallback_name)

        # Fall back if the routed specialist isn't loaded
        if specialist_name not in self.specialists:
            specialist_name = self._fallback_name

        agent = self.specialists[specialist_name]

        # Track routing
        self._routing_counts[specialist_name] = (
            self._routing_counts.get(specialist_name, 0) + 1
        )

        try:
            delta = agent.decide(obs, step)
        except Exception:
            logger.exception(
                "Specialist %s failed, falling back to %s",
                specialist_name, self._fallback_name,
            )
            delta = self.specialists[self._fallback_name].decide(obs, step)

        return delta

    def decide_with_info(self, obs: np.ndarray, step: int) -> dict[str, Any]:
        """Like decide(), but returns detailed routing info.

        Returns
        -------
        dict
            Keys: ``delta``, ``pattern``, ``specialist``, ``confidence``,
            ``features``.
        """
        pattern = self.classifier.update(obs)
        specialist_name = ROUTING_TABLE.get(pattern, self._fallback_name)
        if specialist_name not in self.specialists:
            specialist_name = self._fallback_name

        agent = self.specialists[specialist_name]
        self._routing_counts[specialist_name] = (
            self._routing_counts.get(specialist_name, 0) + 1
        )

        try:
            result = agent.decide_with_info(obs, step)
        except Exception:
            logger.exception("Specialist %s failed", specialist_name)
            result = self.specialists[self._fallback_name].decide_with_info(obs, step)

        result["pattern"] = pattern
        result["specialist"] = specialist_name
        result["confidence"] = self.classifier.confidence
        result["features"] = {
            "cv": self.classifier.features.cv,
            "slope": self.classifier.features.slope,
            "delta_variance": self.classifier.features.delta_variance,
            "spike_ratio": self.classifier.features.spike_ratio,
            "mean_rate": self.classifier.features.mean_rate,
        }

        self._routing_log.append({
            "step": step, "pattern": pattern,
            "specialist": specialist_name,
            "confidence": self.classifier.confidence,
            "delta": result["delta"],
        })

        return result

    def reset(self) -> None:
        """Reset all specialists and the classifier."""
        self.classifier.reset()
        for agent in self.specialists.values():
            agent.reset()
        self._routing_log.clear()

    @property
    def routing_summary(self) -> dict[str, Any]:
        """Summary of routing decisions made so far."""
        total = sum(self._routing_counts.values())
        return {
            name: {
                "count": count,
                "pct": count / max(total, 1) * 100,
            }
            for name, count in sorted(
                self._routing_counts.items(),
                key=lambda x: x[1], reverse=True,
            )
        }

    def __repr__(self) -> str:
        specs = ", ".join(self.specialists.keys())
        return (
            f"EnsembleMetaAgent(specialists=[{specs}], "
            f"window={self.classifier.window_size})"
        )
