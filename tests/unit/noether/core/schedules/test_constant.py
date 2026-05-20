#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schedules import ConstantScheduleConfig
from noether.core.schedules.constant import ConstantSchedule


def test_constant_schedule_initialization():
    """Test initialization of the ConstantSchedule."""
    value = 0.1
    config = ConstantScheduleConfig(value=value)
    schedule = ConstantSchedule(config=config)
    assert schedule.value == value


def test_constant_schedule_get_value():
    """Test the _get_value method of ConstantSchedule."""
    value = 0.05
    total_steps = 100
    config = ConstantScheduleConfig(value=value)
    schedule = ConstantSchedule(config=config)

    assert schedule.get_value(step=0, total_steps=total_steps) == value
    assert schedule.get_value(step=total_steps // 2, total_steps=total_steps) == value
    assert schedule.get_value(step=total_steps - 1, total_steps=total_steps) == value


def test_constant_schedule_str_representation():
    """Test the string representation of ConstantSchedule."""
    value = 0.5
    config = ConstantScheduleConfig(value=value)
    schedule = ConstantSchedule(config=config)
    assert str(schedule) == f"ConstantSchedule(value={value})"
