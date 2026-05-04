#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import math
import unittest

import numpy as np

from noether.core.schedules import SqrtDecreasingSchedule
from noether.core.schemas import DecreasingProgressScheduleConfig


class TestSqrtDecreasingSchedule(unittest.TestCase):
    def test_decreasing(self):
        sched = SqrtDecreasingSchedule(config=DecreasingProgressScheduleConfig(max_value=1.0, end_value=0.0))
        # value(step) = 1 - sqrt(step / (N-1)) for default exclude_first=False, exclude_last=False
        expected = [1.0 - math.sqrt(step / 10) for step in range(11)]
        actual = [sched.get_value(step, total_steps=11) for step in range(11)]
        self.assertTrue(np.allclose(expected, actual), actual)

    def test_decreasing_endpoints(self):
        sched = SqrtDecreasingSchedule(config=DecreasingProgressScheduleConfig(max_value=2.0, end_value=0.5))
        self.assertAlmostEqual(sched.get_value(0, total_steps=11), 2.0)
        self.assertAlmostEqual(sched.get_value(10, total_steps=11), 0.5)

    def test_strictly_decreasing(self):
        sched = SqrtDecreasingSchedule(config=DecreasingProgressScheduleConfig(max_value=1.0, end_value=0.0))
        values = [sched.get_value(step, total_steps=21) for step in range(21)]
        for prev, curr in zip(values[:-1], values[1:]):
            self.assertGreater(prev, curr)

    def test_fast_initial_drop(self):
        # Hägele 1-sqrt cooldown distinguisher: sqrt drops faster than linear/cosine in the first half.
        # At 25% progress: sqrt(0.25) = 0.5, so value = 1 - 0.5 = 0.5. Linear would be 1 - 0.25 = 0.75.
        sched = SqrtDecreasingSchedule(config=DecreasingProgressScheduleConfig(max_value=1.0, end_value=0.0))
        self.assertAlmostEqual(sched.get_value(5, total_steps=21), 1.0 - math.sqrt(5 / 20))
