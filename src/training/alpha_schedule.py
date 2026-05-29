"""
AlphaScheduleCallback — Cyclically anneal the cost-SLA tradeoff during training.

This callback adjusts the `alpha` parameter on all parallel environments
during training, implementing a curriculum where the agent:
  1. First learns safe scaling (SLA-focused, low alpha)
  2. Then learns cost efficiency (higher alpha)
  3. Finally stabilizes at a balanced tradeoff

Usage:
    schedule = [(0, 0.1), (200_000, 0.1), (500_000, 0.4), (700_000, 0.2)]
    cb = AlphaScheduleCallback(vec_env, schedule)
    model.learn(callback=cb)
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class AlphaScheduleCallback(BaseCallback):
    """Cyclically anneal the cost-SLA tradeoff weight alpha.

    Parameters
    ----------
    venv : VecEnv
        The vectorized environment (must wrap K8sSimEnv instances with set_alpha).
    schedule : list[tuple[int, float]]
        List of (timestep, alpha) breakpoints. Alpha is linearly
        interpolated between consecutive breakpoints.
    verbose : int
        Verbosity level.
    """

    def __init__(
        self,
        venv: VecEnv,
        schedule: list[tuple[int, float]],
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.venv = venv
        self.schedule = sorted(schedule, key=lambda x: x[0])

    def _on_step(self) -> bool:
        alpha = self._interpolate(self.num_timesteps)
        self.venv.env_method("set_alpha", alpha)

        # Log alpha to TensorBoard periodically
        if self.verbose and self.num_timesteps % 10000 == 0:
            self.logger.record("train/alpha", alpha)

        return True

    def _interpolate(self, t: int) -> float:
        """Linearly interpolate alpha from the schedule breakpoints."""
        if t <= self.schedule[0][0]:
            return self.schedule[0][1]

        for i in range(len(self.schedule) - 1):
            t0, a0 = self.schedule[i]
            t1, a1 = self.schedule[i + 1]
            if t0 <= t < t1:
                frac = (t - t0) / max(t1 - t0, 1)
                return a0 + frac * (a1 - a0)

        return self.schedule[-1][1]
