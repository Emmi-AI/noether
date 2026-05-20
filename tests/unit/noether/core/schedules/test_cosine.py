#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import unittest

import numpy as np

from noether.core.schedules import (
    CosineDecreasingSchedule,
    CosineIncreasingSchedule,
    DecreasingProgressScheduleConfig,
    IncreasingProgressScheduleConfig,
)


class TestCosineDecreasingSchedule(unittest.TestCase):
    def test_decreasing(self):
        sched = CosineDecreasingSchedule(config=DecreasingProgressScheduleConfig(max_value=1.0, end_value=0.0))
        expected = [
            1.0,
            0.9755282581475768,
            0.9045084971874737,
            0.7938926261462366,
            0.6545084971874737,
            0.5,
            0.3454915028125263,
            0.20610737385376354,
            0.09549150281252627,
            0.02447174185242318,
            0.0,
        ]
        actual = [sched.get_value(step, total_steps=11) for step in range(11)]
        self.assertTrue(np.allclose(expected, actual), actual)

    def test_decreasing_overhang(self):
        sched = CosineDecreasingSchedule(config=DecreasingProgressScheduleConfig(max_value=1.0, end_value=0.0))
        sched_percent = CosineDecreasingSchedule(
            config=DecreasingProgressScheduleConfig(max_value=1.0, end_value=0.0, overhang_percent=0.25)
        )
        sched_steps = CosineDecreasingSchedule(
            config=DecreasingProgressScheduleConfig(max_value=1.0, end_value=0.0, overhang_steps=2)
        )
        expected = [
            1.0,
            0.9829629131445341,
            0.9330127018922194,
            0.8535533905932737,
            0.75,
            0.6294095225512604,
            0.5,
            0.37059047744873963,
            0.2500000000000001,
            0.14644660940672627,
            0.06698729810778059,
        ]
        actual = [sched.get_value(step, total_steps=11 + int(11 * 0.25)) for step in range(11)]
        actual_percent = [sched_percent.get_value(step, total_steps=11) for step in range(11)]
        actual_steps = [sched_steps.get_value(step, total_steps=11) for step in range(11)]
        self.assertTrue(np.allclose(expected, actual), actual)
        self.assertTrue(np.allclose(expected, actual_percent), actual_percent)
        self.assertTrue(np.allclose(expected, actual_steps), actual_steps)


class TestCosineIncreasingSchedule(unittest.TestCase):
    def test_increasing(self):
        sched = CosineIncreasingSchedule(config=IncreasingProgressScheduleConfig(max_value=1.0, start_value=0.0))
        expected = [
            0.0,
            0.024471741852423234,
            0.09549150281252633,
            0.2061073738537635,
            0.34549150281252633,
            0.5,
            0.6545084971874737,
            0.7938926261462366,
            0.9045084971874737,
            0.9755282581475768,
            1.0,
        ]
        actual = [sched.get_value(step, total_steps=11) for step in range(11)]
        self.assertTrue(np.allclose(expected, actual), actual)

    def test_increasing_absmin(self):
        sched = CosineIncreasingSchedule(config=IncreasingProgressScheduleConfig(max_value=1.0, start_value=1e-5))
        expected = [
            1e-05,
            0.02448149713500471,
            0.0955005478974982,
            0.20611531278002496,
            0.3454980478974982,
            0.500005,
            0.6545119521025018,
            0.7938946872199751,
            0.9045094521025019,
            0.9755285028649954,
            1.0,
        ]
        actual = [sched.get_value(step, total_steps=11) for step in range(11)]
        self.assertTrue(np.allclose(expected, actual), actual)
