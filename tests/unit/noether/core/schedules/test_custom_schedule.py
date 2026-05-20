#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import unittest

import numpy as np

from noether.core.schedules import CustomSchedule, CustomScheduleConfig


class TestCustomSchedule(unittest.TestCase):
    def test(self):
        rng = np.random.default_rng(82934)
        values = rng.random(size=(11,)).tolist()
        sched = CustomSchedule(config=CustomScheduleConfig(values=values))
        actual = [sched.get_value(step, total_steps=11) for step in range(11)]
        self.assertTrue(np.allclose(values, actual), actual)
