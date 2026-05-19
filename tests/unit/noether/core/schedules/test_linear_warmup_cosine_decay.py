#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import unittest

import numpy as np

from noether.core.schedules import LinearWarmupCosineDecaySchedule, LinearWarmupCosineDecayScheduleConfig


class TestLinearWarmupCosineDecaySchedule(unittest.TestCase):
    def test_factory_percent(self):
        schedule = LinearWarmupCosineDecaySchedule(
            LinearWarmupCosineDecayScheduleConfig(warmup_percent=0.2, max_value=1.0, end_value=1e-6)
        )

        expected = [
            0.3333333333333333,
            0.6666666666666666,
            1.0,
            0.9619398043158771,
            0.8535535370398831,
            0.6913420248408287,
            0.5000005000000001,
            0.3086589751591714,
            0.1464474629601169,
            0.03806119568412292,
            1e-06,
        ]
        actual = [schedule.get_value(step, total_steps=11) for step in range(11)]
        self.assertTrue(np.allclose(expected, actual), actual)

    def test_factory_steps(self):
        schedule = LinearWarmupCosineDecaySchedule(
            LinearWarmupCosineDecayScheduleConfig(warmup_steps=2, max_value=1.0, end_value=1e-6)
        )

        expected = [
            0.3333333333333333,
            0.6666666666666666,
            1.0,
            0.9619398043158771,
            0.8535535370398831,
            0.6913420248408287,
            0.5000005000000001,
            0.3086589751591714,
            0.1464474629601169,
            0.03806119568412292,
            1e-06,
        ]
        actual = [schedule.get_value(step, total_steps=11) for step in range(11)]
        self.assertTrue(np.allclose(expected, actual), actual)
