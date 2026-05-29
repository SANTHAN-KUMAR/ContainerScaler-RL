"""
WorkloadGenerator — Synthetic traffic patterns for simulator training.

Generates one of 5 traffic shapes per episode:
  steady, diurnal, flash_crowd, gradual_ramp, noisy

Called by K8sSimEnv on every step to get current request rate (rps).
"""

from __future__ import annotations

import numpy as np

# Training patterns — used during RL training
PATTERNS = ("steady", "diurnal", "flash_crowd", "gradual_ramp", "noisy")

# Held-out patterns — NEVER used in training, only for evaluation.
# If the agent degrades >10% on these, it has overfit to the training patterns.
HELD_OUT_PATTERNS = ("double_peak", "sawtooth")

# All patterns combined (for evaluation scripts)
ALL_PATTERNS = PATTERNS + HELD_OUT_PATTERNS


class WorkloadGenerator:
    """Generates synthetic traffic (rps) at each simulation step.

    Parameters
    ----------
    pattern : str
        One of the 5 named patterns, or ``"random"`` to pick one
        uniformly at random on each ``reset()`` call.
    seed : int | None
        RNG seed for reproducible noise.
    """

    def __init__(self, pattern: str = "random", seed: int | None = None) -> None:
        self._requested_pattern = pattern
        self._rng = np.random.default_rng(seed)

        # Active pattern for this episode (set by reset)
        self.pattern: str = ""
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_rate(self, step: int) -> float:
        """Return the request rate (rps) for the given step (0–119).

        The output is always clamped to a minimum of 1.0 rps to avoid
        division-by-zero in queue dynamics.
        """
        t = step / 119.0  # normalized time in [0, 1]
        rate = self._dispatch(step, t)
        return float(max(1.0, rate))

    def reset(self) -> None:
        """Select a (possibly random) pattern for the next episode."""
        if self._requested_pattern == "random":
            self.pattern = self._rng.choice(PATTERNS)
        elif self._requested_pattern == "held_out_random":
            self.pattern = self._rng.choice(HELD_OUT_PATTERNS)
        else:
            if self._requested_pattern not in ALL_PATTERNS:
                raise ValueError(
                    f"Unknown pattern '{self._requested_pattern}'. "
                    f"Choose from {ALL_PATTERNS}, 'random', or 'held_out_random'."
                )
            self.pattern = self._requested_pattern

    # ------------------------------------------------------------------
    # Pattern implementations
    # ------------------------------------------------------------------

    def _dispatch(self, step: int, t: float) -> float:
        """Route to the active pattern function."""
        if self.pattern == "steady":
            return self._steady()
        if self.pattern == "diurnal":
            return self._diurnal(step)
        if self.pattern == "flash_crowd":
            return self._flash_crowd(step)
        if self.pattern == "gradual_ramp":
            return self._gradual_ramp(t)
        if self.pattern == "noisy":
            return self._noisy()
        if self.pattern == "double_peak":
            return self._double_peak(step)
        if self.pattern == "sawtooth":
            return self._sawtooth(step)
        raise ValueError(f"Invalid pattern: {self.pattern}")

    def _steady(self) -> float:
        """Flat ~100 rps + Gaussian noise (σ=10)."""
        return 100.0 + self._rng.normal(0.0, 10.0)

    def _diurnal(self, step: int) -> float:
        """Sine wave 50→200→50 rps over one episode."""
        return 125.0 + 75.0 * np.sin(2.0 * np.pi * step / 120.0)

    def _flash_crowd(self, step: int) -> float:
        """Baseline 100, spike to 450 at ~30% into the episode."""
        noise = self._rng.normal(0.0, 5.0)
        if step < 36:
            return 100.0 + noise
        if step < 48:
            # Ramp up over 12 steps (36→48)
            progress = (step - 36) / 12.0
            return 100.0 + 350.0 * progress + noise
        if step < 72:
            return 450.0 + noise
        # Decay back to baseline over remaining steps
        progress = (step - 72) / (119 - 72)
        return 450.0 - 350.0 * progress + noise

    def _gradual_ramp(self, t: float) -> float:
        """Linear ramp 50→450 rps over the full episode."""
        noise = self._rng.normal(0.0, 5.0)
        return 50.0 + 400.0 * t + noise

    def _noisy(self) -> float:
        """Log-normal traffic (mean≈100, σ=0.7) — unpredictable."""
        return float(self._rng.lognormal(mean=np.log(100.0), sigma=0.7))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"WorkloadGenerator(pattern={self.pattern!r})"

    # ------------------------------------------------------------------
    # Held-out patterns (evaluation only — never used in training)
    # ------------------------------------------------------------------

    def _double_peak(self, step: int) -> float:
        """Two sine peaks at t=0.25 and t=0.75 — tests multi-modal anticipation.

        Peak 1: ~350 rps at step 30, Peak 2: ~350 rps at step 90.
        """
        noise = self._rng.normal(0.0, 5.0)
        base = 100.0
        peak = 250.0 * (
            np.sin(2.0 * np.pi * step / 60.0) ** 2  # two peaks per episode
        )
        return base + peak + noise

    def _sawtooth(self, step: int) -> float:
        """Linear ramp 50→300, instant drop to 50, repeat 3× — tests rapid recovery.

        Each cycle is 40 steps. Ramp for 38 steps, then instant drop.
        """
        noise = self._rng.normal(0.0, 5.0)
        cycle_length = 40
        pos_in_cycle = step % cycle_length
        ramp_progress = pos_in_cycle / (cycle_length - 2)
        rate = 50.0 + 250.0 * min(ramp_progress, 1.0)
        return rate + noise
