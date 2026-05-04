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
            WarmupStableDecayScheduleConfig(
                max_value=1.0, warmup_percent=0.02, warmup_steps=100, cooldown_percent=0.1
            )

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
            WarmupStableDecayScheduleConfig(
                max_value=1.0, warmup_percent=0.02, cooldown_steps=100
            )

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
