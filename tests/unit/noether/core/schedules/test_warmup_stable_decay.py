#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import unittest

import pytest

from noether.core.schemas import WarmupStableDecayScheduleConfig


class TestWarmupStableDecayScheduleConfigValidation(unittest.TestCase):
    def test_accepts_percent_mode(self):
        cfg = WarmupStableDecayScheduleConfig(
            max_value=1.0, warmup_percent=0.02, cooldown_percent=0.1, cooldown_shape="sqrt"
        )
        self.assertEqual(cfg.cooldown_shape, "sqrt")

    def test_accepts_steps_mode(self):
        cfg = WarmupStableDecayScheduleConfig(
            max_value=1.0, warmup_steps=100, cooldown_steps=200, cooldown_shape="linear"
        )
        self.assertEqual(cfg.cooldown_shape, "linear")

    def test_rejects_warmup_both_percent_and_steps(self):
        with pytest.raises(ValueError, match="warmup_steps or warmup_percent"):
            WarmupStableDecayScheduleConfig(max_value=1.0, warmup_percent=0.02, warmup_steps=100, cooldown_percent=0.1)

    def test_rejects_cooldown_both_percent_and_steps(self):
        with pytest.raises(ValueError, match="cooldown_steps or cooldown_percent"):
            WarmupStableDecayScheduleConfig(
                max_value=1.0, warmup_percent=0.02, cooldown_percent=0.1, cooldown_steps=100
            )

    def test_rejects_warmup_neither(self):
        with pytest.raises(ValueError, match="warmup_steps or warmup_percent"):
            WarmupStableDecayScheduleConfig(max_value=1.0, cooldown_percent=0.1)

    def test_rejects_cooldown_neither(self):
        with pytest.raises(ValueError, match="cooldown_steps or cooldown_percent"):
            WarmupStableDecayScheduleConfig(max_value=1.0, warmup_percent=0.02)

    def test_rejects_mixing_percent_and_steps_modes(self):
        with pytest.raises(ValueError, match="both use percent or both use steps"):
            WarmupStableDecayScheduleConfig(max_value=1.0, warmup_percent=0.02, cooldown_steps=100)

    def test_rejects_warmup_plus_cooldown_exceeding_one(self):
        with pytest.raises(ValueError, match=r"warmup_percent \+ cooldown_percent"):
            WarmupStableDecayScheduleConfig(max_value=1.0, warmup_percent=0.6, cooldown_percent=0.5)

    def test_rejects_invalid_cooldown_shape(self):
        with pytest.raises(ValueError):
            WarmupStableDecayScheduleConfig(
                max_value=1.0, warmup_percent=0.02, cooldown_percent=0.1, cooldown_shape="exponential"
            )

    def test_default_cooldown_shape_is_sqrt(self):
        cfg = WarmupStableDecayScheduleConfig(max_value=1.0, warmup_percent=0.02, cooldown_percent=0.1)
        self.assertEqual(cfg.cooldown_shape, "sqrt")


import numpy as np

from noether.core.schedules import WarmupStableDecaySchedule


class TestWarmupStableDecayScheduleBehavior(unittest.TestCase):
    """Phase-boundary and shape-distinction tests."""

    # Use N=21 with warmup_percent=0.1, cooldown_percent=0.1 so phases are clearly delimited.
    # warmup_end = int(0.1 * 20) = 2 -> warmup phase: steps 0, 1
    # cooldown_start = int(0.9 * 20) = 18 -> stable phase: steps 2..17, cooldown: 18, 19, 20

    N = 21
    WARMUP_END = 2
    COOLDOWN_START = 18

    def _make(self, shape: str) -> WarmupStableDecaySchedule:
        return WarmupStableDecaySchedule(
            WarmupStableDecayScheduleConfig(
                max_value=1.0, end_value=0.0, warmup_percent=0.1, cooldown_percent=0.1, cooldown_shape=shape
            )
        )

    def test_warmup_monotone_increasing(self):
        sched = self._make("sqrt")
        warmup_vals = [sched.get_value(s, self.N) for s in range(self.WARMUP_END)]
        for prev, curr in zip(warmup_vals[:-1], warmup_vals[1:], strict=True):
            self.assertLess(prev, curr)

    def test_stable_phase_at_max(self):
        for shape in ("sqrt", "linear", "cosine"):
            sched = self._make(shape)
            for s in range(self.WARMUP_END, self.COOLDOWN_START):
                self.assertAlmostEqual(sched.get_value(s, self.N), 1.0, msg=f"shape={shape} step={s}")

    def test_cooldown_monotone_decreasing(self):
        for shape in ("sqrt", "linear", "cosine"):
            sched = self._make(shape)
            cooldown_vals = [sched.get_value(s, self.N) for s in range(self.COOLDOWN_START, self.N)]
            for prev, curr in zip(cooldown_vals[:-1], cooldown_vals[1:], strict=True):
                self.assertGreater(prev, curr, msg=f"shape={shape}")

    def test_cooldown_endpoint_reaches_end_value(self):
        for shape in ("sqrt", "linear", "cosine"):
            sched = self._make(shape)
            self.assertAlmostEqual(sched.get_value(self.N - 1, self.N), 0.0, msg=f"shape={shape}")

    def test_sqrt_cooldown_drops_faster_than_linear(self):
        # At 50% through the cooldown phase, sqrt should be strictly below linear (sqrt(0.5) > 0.5).
        # cooldown spans steps 18..20 (3 values). step 19 is the midpoint.
        sqrt_v = self._make("sqrt").get_value(19, self.N)
        linear_v = self._make("linear").get_value(19, self.N)
        cosine_v = self._make("cosine").get_value(19, self.N)
        # sqrt drops fastest, then linear == cosine at midpoint exactly
        self.assertLess(sqrt_v, linear_v)
        self.assertAlmostEqual(linear_v, cosine_v)

    def test_steps_mode_matches_percent_mode(self):
        # With N=21 and percent (0.1, 0.1), warmup_end=2 and cooldown_start=18 (cooldown len=3).
        # The equivalent steps-mode config: warmup_steps=2, cooldown_steps=3.
        pct = WarmupStableDecaySchedule(
            WarmupStableDecayScheduleConfig(
                max_value=1.0, end_value=0.0, warmup_percent=0.1, cooldown_percent=0.1, cooldown_shape="sqrt"
            )
        )
        steps = WarmupStableDecaySchedule(
            WarmupStableDecayScheduleConfig(
                max_value=1.0, end_value=0.0, warmup_steps=2, cooldown_steps=3, cooldown_shape="sqrt"
            )
        )
        pct_vals = [pct.get_value(s, self.N) for s in range(self.N)]
        steps_vals = [steps.get_value(s, self.N) for s in range(self.N)]
        self.assertTrue(np.allclose(pct_vals, steps_vals), (pct_vals, steps_vals))
