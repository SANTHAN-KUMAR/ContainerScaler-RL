"""Tests for WorkloadGenerator."""

from __future__ import annotations

import numpy as np
import pytest

from src.env.workload import (
    ALL_PATTERNS, HELD_OUT_PATTERNS, PATTERNS, WorkloadGenerator,
)


class TestWorkloadPatterns:
    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_all_patterns_produce_positive_rates(self, pattern):
        wg = WorkloadGenerator(pattern=pattern, seed=42)
        for step in range(120):
            rate = wg.get_rate(step)
            assert rate >= 1.0, f"{pattern} produced rate {rate} at step {step}"

    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_all_patterns_produce_reasonable_rates(self, pattern):
        wg = WorkloadGenerator(pattern=pattern, seed=42)
        rates = [wg.get_rate(step) for step in range(120)]
        assert max(rates) < 2000, f"{pattern}: max rate {max(rates)} is unreasonably high"

    def test_random_pattern_picks_from_training(self):
        wg = WorkloadGenerator(pattern="random", seed=42)
        patterns_seen = set()
        for _ in range(200):
            wg.reset()
            patterns_seen.add(wg.pattern)
        # Should see all training patterns eventually
        assert patterns_seen.issubset(set(PATTERNS))
        assert len(patterns_seen) >= 3  # at least 3 of 5

    def test_held_out_random(self):
        wg = WorkloadGenerator(pattern="held_out_random", seed=42)
        for _ in range(50):
            wg.reset()
            assert wg.pattern in HELD_OUT_PATTERNS

    def test_invalid_pattern_raises(self):
        with pytest.raises(ValueError, match="Unknown pattern"):
            WorkloadGenerator(pattern="nonexistent", seed=42)


class TestReproducibility:
    def test_same_seed_same_rates(self):
        wg1 = WorkloadGenerator(pattern="noisy", seed=42)
        wg2 = WorkloadGenerator(pattern="noisy", seed=42)
        for step in range(120):
            assert wg1.get_rate(step) == wg2.get_rate(step)

    def test_different_seeds_different_rates(self):
        wg1 = WorkloadGenerator(pattern="noisy", seed=42)
        wg2 = WorkloadGenerator(pattern="noisy", seed=99)
        rates1 = [wg1.get_rate(s) for s in range(120)]
        rates2 = [wg2.get_rate(s) for s in range(120)]
        assert rates1 != rates2


class TestPatternCharacteristics:
    def test_steady_is_around_100(self):
        wg = WorkloadGenerator(pattern="steady", seed=42)
        rates = [wg.get_rate(s) for s in range(120)]
        assert 70 < np.mean(rates) < 130

    def test_diurnal_has_sinusoidal_shape(self):
        wg = WorkloadGenerator(pattern="diurnal", seed=42)
        rates = [wg.get_rate(s) for s in range(120)]
        assert min(rates) < 80  # should dip low
        assert max(rates) > 180  # should peak high

    def test_flash_crowd_has_spike(self):
        wg = WorkloadGenerator(pattern="flash_crowd", seed=42)
        rates = [wg.get_rate(s) for s in range(120)]
        assert max(rates) > 350  # should have a major spike

    def test_gradual_ramp_increases(self):
        wg = WorkloadGenerator(pattern="gradual_ramp", seed=42)
        early = np.mean([wg.get_rate(s) for s in range(10)])
        wg2 = WorkloadGenerator(pattern="gradual_ramp", seed=42)
        late = np.mean([wg2.get_rate(s) for s in range(110, 120)])
        assert late > early * 2


class TestConstants:
    def test_patterns_tuple(self):
        assert len(PATTERNS) == 5
        assert "steady" in PATTERNS
        assert "diurnal" in PATTERNS

    def test_held_out_separate(self):
        for p in HELD_OUT_PATTERNS:
            assert p not in PATTERNS

    def test_all_patterns_combined(self):
        assert ALL_PATTERNS == PATTERNS + HELD_OUT_PATTERNS
