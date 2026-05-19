#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import unittest

import numpy as np
from pydantic import ValidationError

from noether.core.schedules import (
    StepFixedSchedule,
    StepFixedScheduleConfig,
    StepIntervalSchedule,
    StepIntervalScheduleConfig,
)


class TestStepFixedSchedule(unittest.TestCase):
    def test_invalid_steps(self):
        with self.assertRaises(ValidationError):
            StepFixedSchedule(StepFixedScheduleConfig(start_value=1.0, factor=0.5, steps=None))
        with self.assertRaises(ValidationError):
            StepFixedSchedule(StepFixedScheduleConfig(start_value=1.0, factor=0.5, steps=[]))
        with self.assertRaises(ValidationError):
            StepFixedSchedule(StepFixedScheduleConfig(start_value=1.0, factor=0.5, steps=[5]))
        with self.assertRaises(ValidationError):
            StepFixedSchedule(StepFixedScheduleConfig(start_value=1.0, factor=0.5, steps=[-1]))

    def test_increasing(self):
        sched = StepFixedSchedule(StepFixedScheduleConfig(start_value=1.0, factor=1.5, steps=[0.2, 0.5, 0.8]))
        actual = [sched.get_value(step, total_steps=10) for step in range(10)]
        self.assertTrue(np.allclose([1.0, 1.0, 1.5, 1.5, 1.5, 2.25, 2.25, 2.25, 3.375, 3.375], actual))

    def test_decreasing(self):
        sched = StepFixedSchedule(StepFixedScheduleConfig(start_value=1.0, factor=0.5, steps=[0.2, 0.5, 0.8]))
        actual = [sched.get_value(step, total_steps=10) for step in range(10)]
        self.assertTrue(np.allclose([1.0, 1.0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.125, 0.125], actual))


class TestStepIntervalSchedule(unittest.TestCase):
    def test_invalid_interval(self):
        with self.assertRaises(ValidationError):
            StepIntervalSchedule(StepIntervalScheduleConfig(start_value=1.0, factor=0.5, update_interval=0.0))
        with self.assertRaises(ValidationError):
            StepIntervalSchedule(StepIntervalScheduleConfig(start_value=1.0, factor=0.5, update_interval=1.0))

    def test_increasing(self):
        sched = StepIntervalSchedule(StepIntervalScheduleConfig(start_value=1.0, factor=1.5, update_interval=0.2))
        actual = [sched.get_value(step, total_steps=10) for step in range(10)]
        self.assertTrue(np.allclose([1.0, 1.0, 1.5, 1.5, 2.25, 2.25, 3.375, 3.375, 5.0625, 5.0625], actual))

    def test_decreasing(self):
        sched = StepIntervalSchedule(StepIntervalScheduleConfig(start_value=1.0, factor=0.5, update_interval=0.2))
        actual = [sched.get_value(step, total_steps=10) for step in range(10)]
        self.assertTrue(np.allclose([1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.125, 0.125, 0.0625, 0.0625], actual))
