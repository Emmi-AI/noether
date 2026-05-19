#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
import pytest

from noether.core.schedules import (
    ConstantSchedule,
    ConstantScheduleConfig,
    IncreasingProgressScheduleConfig,
    LinearIncreasingSchedule,
)
from noether.core.utils.training import ScheduleWrapper, TrainingIteration, UpdateCounter


class TestScheduleWrapper:
    def return_update_counter(self):
        UpdateCounter(
            updates_per_epoch=10,
            start_iteration=TrainingIteration(epoch=0, update=0, sample=0),
            end_iteration=TrainingIteration(epoch=10),
            effective_batch_size=1,
        )

    def return_constant_schedule(self):
        return ConstantSchedule(ConstantScheduleConfig(value=0.1))

    def test_init_assigns_attributes(self):
        # Arrange
        schedule = self.return_constant_schedule()
        update_counter = self.return_update_counter()
        interval = "update"

        # Act
        wrapper = ScheduleWrapper(schedule=schedule, update_counter=update_counter, interval=interval)

        # Assert
        assert wrapper.schedule is schedule
        assert wrapper.update_counter is update_counter
        assert wrapper.interval == interval

    def test_init_default_values(self):
        # Arrange
        schedule = self.return_constant_schedule()

        # Act
        wrapper = ScheduleWrapper(schedule=schedule)

        # Assert
        assert wrapper.schedule is schedule
        assert wrapper.update_counter is None
        assert wrapper.interval == "update"

    def test_init_with_epoch_interval(self):
        # Arrange
        schedule = self.return_constant_schedule()
        update_counter = self.return_update_counter()
        interval = "epoch"

        # Act
        wrapper = ScheduleWrapper(schedule=schedule, update_counter=update_counter, interval=interval)

        # Assert
        assert wrapper.schedule is schedule
        assert wrapper.update_counter is update_counter
        assert wrapper.interval == interval

        for i in range(0, 10):
            value = wrapper.get_value()
            assert value == 0.1

    def test_init_raises_with_non_constant_schedule_and_no_update_counter(self):
        with pytest.raises(NotImplementedError):
            ScheduleWrapper(
                schedule=self.return_constant_schedule(), update_counter=self.return_update_counter(), interval="test"
            )

    def test_linear_scheduler_wrapper(self):
        # Arrange
        updates = 10.0
        schedule_config = IncreasingProgressScheduleConfig(
            start_value=0.0,
            max_value=1.0,
        )
        schedule = LinearIncreasingSchedule(schedule_config)
        update_counter = UpdateCounter(
            updates_per_epoch=10,
            start_iteration=TrainingIteration(epoch=0, update=0, sample=0),
            end_iteration=TrainingIteration(update=updates),
            effective_batch_size=1,
        )
        interval = "update"

        # Act
        wrapper = ScheduleWrapper(schedule=schedule, update_counter=update_counter, interval=interval)

        # Assert
        assert wrapper.schedule is schedule
        assert wrapper.update_counter is update_counter
        assert wrapper.interval == interval

        # Simulate progress from 0 to 1
        expected = [
            0.0,
            0.1111111111111111,
            0.2222222222222222,
            0.3333333333333333,
            0.4444444444444444,
            0.5555555555555556,
            0.6666666666666666,
            0.7777777777777778,
            0.8888888888888888,
            1.0,
        ]
        for i in range(10):
            value = wrapper.get_value()
            expected_value = i / updates  # Since we have 10 updates to reach from 0.0 to 1.0
            update_counter.next_update()
            assert value == expected[i]
