#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import abc
from collections.abc import Sequence
from dataclasses import dataclass

from noether.core.schedules.schemas import (
    DecreasingProgressScheduleConfig,
    IncreasingProgressScheduleConfig,
    ProgressScheduleConfig,
)


class ScheduleBase:
    """Base class for schedules."""

    def __init__(self, overhang_percent: float | None = None, overhang_steps: int | None = None):
        """Initialize the scheduler.

        Args:
            overhang_percent: The percentage by which the schedule is artificially prolonged.
            overhang_steps: The number of steps by which the schedule is artificially prolonged.
        """

        self.overhang_percent = overhang_percent
        self.overhang_steps = overhang_steps

        if self.overhang_percent is not None and self.overhang_steps is not None:
            raise ValueError("only one of overhang_percent and overhang_steps must be set")

        # check that correct method is overwritten
        if not type(self).get_value == ScheduleBase.get_value:
            raise NotImplementedError(
                "The '_get_value' method must be implemented by subclasses instead of overriding 'get_value'."
            )

    def get_value(self, step: int, total_steps: int) -> float:
        """Get the value of the schedule at a given step.

        This function includes the correction for overhangs in percent or steps and then calls the _get_value method.
        Therefore, it should not be overwritten by subclasses. Instead, the _get_value method should be implemented.

        Args:
            step: The step for which to get the scheduler value.
            total_steps: The total number of steps.
        """
        if not (0 <= step <= total_steps):
            raise ValueError(f"0 <= step <= total_steps (step={step} total_steps={total_steps})")
        if self.overhang_percent is not None:
            total_steps += int(total_steps * self.overhang_percent)
        if self.overhang_steps is not None:
            total_steps += self.overhang_steps
        return self._get_value(step, total_steps)

    @abc.abstractmethod
    def _get_value(self, step: int, total_steps: int) -> float:
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    def __str__(self):
        return type(self).__name__


class ProgressSchedule(ScheduleBase):
    """Base class for schedules that progress monotonically in value over time."""

    def __init__(
        self,
        config: ProgressScheduleConfig,
        start_value: float,
        delta: float,
    ):
        """Initialize the scheduler.

        Args:
            config: Configuration for the progress schedule.
            start_value: The initial value of the scheduler.
            delta: The total change in value over the schedule.
        """
        super().__init__(overhang_percent=config.overhang_percent, overhang_steps=config.overhang_steps)

        self.start_value = start_value
        self.delta = delta
        self.exclude_first = config.exclude_first
        self.exclude_last = config.exclude_last

    def __str__(self):
        return (
            f"{type(self).__name__}("
            f"start={self.start_value}, "
            f"end={self.start_value + self.delta}, "
            f"excl_first={self.exclude_first}, "
            f"excl_last={self.exclude_last}"
            ")"
        )

    def _get_value(self, step: int, total_steps: int) -> float:
        if self.exclude_last:
            total_steps += 1
        if self.exclude_first:
            step += 1
            total_steps += 1
        # get progress of schedule (going from 0 to 1)
        progress = self._get_progress(step, total_steps)
        # adjust to "absolute value" (i.e. real learning rate)
        return self.start_value + progress * self.delta

    def _get_progress(self, step: int, total_steps: int) -> float:
        raise NotImplementedError


class DecreasingProgressSchedule(ProgressSchedule):
    """Base class for schedules that monotonically decrease in value over time."""

    def __init__(self, config: DecreasingProgressScheduleConfig):
        """Initialize the scheduler."""
        delta = config.end_value - config.max_value
        if delta > 0.0:
            raise ValueError("end_value must be less than or equal to max_value")
        super().__init__(start_value=config.max_value, delta=delta, config=config)

    def _get_progress(self, step: int, total_steps: int) -> float:
        raise NotImplementedError


class IncreasingProgressSchedule(ProgressSchedule):
    """Base class for schedules that monotonically increase in value over time."""

    def __init__(self, config: IncreasingProgressScheduleConfig):
        """Initialize the scheduler."""
        assert config.max_value is not None
        delta = config.max_value - config.start_value
        assert delta >= 0.0
        super().__init__(start_value=config.start_value, delta=delta, config=config)

    def _get_progress(self, step: int, total_steps: int) -> float:
        raise NotImplementedError


@dataclass
class SequentialPercentScheduleConfig:
    schedule: ScheduleBase
    start_percent: float | None = None
    end_percent: float | None = None

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.start_percent} - {self.end_percent} {self.schedule}"


class SequentialPercentSchedule(ScheduleBase):
    """A scheduler that switches between multiple schedules based on the percentage of steps completed."""

    def __init__(self, schedule_configs: Sequence[SequentialPercentScheduleConfig]):
        """Initialize the scheduler.

        Args:
            schedule_configs: A list of schedule configurations.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__()
        assert len(schedule_configs) > 0
        self.schedule_configs = schedule_configs

        # if first schedule has no start_percent -> set to 0 -> this ensures every schedule has a start_percent
        if schedule_configs[0].start_percent is None:
            schedule_configs[0].start_percent = 0.0
        # propagate start/end
        for i in range(1, len(schedule_configs) - 1):
            # take start_percent from previous schedule
            cur_config = schedule_configs[i]
            if cur_config.start_percent is None:
                next_config = schedule_configs[i - 1]
                assert next_config.end_percent is not None
                cur_config.start_percent = next_config.end_percent
            # take end_percent from next schedule
            if cur_config.end_percent is None:
                cur_config.end_percent = schedule_configs[i + 1].start_percent
        # edge case: last schedule propagate
        if len(schedule_configs) > 1:
            # propagate [-2].end_percent forward
            if schedule_configs[-1].start_percent is None:
                schedule_configs[-1].start_percent = schedule_configs[-2].end_percent
            # propagate [-1].start_percent backward
            if schedule_configs[-2].end_percent is None:
                schedule_configs[-2].end_percent = schedule_configs[-1].start_percent
        # set [-1].end_percent to 1.
        if schedule_configs[-1].end_percent is None:
            schedule_configs[-1].end_percent = 1.0

        # check correctness of start/end
        if len(schedule_configs) == 1:
            # edge case: single schedule
            # always: 0. <= start <= 1.
            # if end is not None: start <= end <= 1.
            start = schedule_configs[0].start_percent
            end = schedule_configs[0].end_percent
            assert start is not None and 0.0 <= start <= 1.0
            if end is not None:
                assert start <= end <= 1.0
        else:
            # check 0 <= cfg[i].start <= cfg[i].end <= 1.
            # check cfg[i].end <= cfg[i+1].start <= 1.
            for i in range(len(schedule_configs) - 1):
                cur_start = schedule_configs[i].start_percent
                cur_end = schedule_configs[i].end_percent
                next_start = schedule_configs[i + 1].start_percent
                assert cur_start is not None and cur_end is not None
                assert 0.0 <= cur_start <= cur_end <= 1.0
                assert next_start is not None and cur_end <= next_start <= 1.0
            # last schedule is allowed to have no end_percent
            last_start = schedule_configs[-1].start_percent
            last_end = schedule_configs[-1].end_percent
            assert last_start is not None
            if last_end is None:
                assert 0.0 <= last_start <= 1.0
            else:
                assert 0.0 <= last_start <= last_end <= 1.0

    def get_sequential_schedule_config(self, step: int, total_steps: int) -> SequentialPercentScheduleConfig | None:
        total_steps_m1 = total_steps - 1
        # percent < config[0].start_percent -> None
        # config[-1].end_percent < percent -> config[-1]
        for i in reversed(range(len(self.schedule_configs))):
            start_pct = self.schedule_configs[i].start_percent
            if start_pct is not None and int(start_pct * total_steps_m1) <= step:
                return self.schedule_configs[i]
        return None

    def _get_value(self, step: int, total_steps: int) -> float:
        config = self.get_sequential_schedule_config(step, total_steps)
        total_steps_m1 = total_steps - 1
        if config is None:
            # adjust step/total_steps within SequentialSchedule to step/total_steps within schedule
            start_pct = self.schedule_configs[0].start_percent
            end_pct = self.schedule_configs[0].end_percent
            assert start_pct is not None
            adj_step = int(total_steps_m1 * start_pct)
            if end_pct == 1.0:
                end_step = total_steps
            else:
                end_step = int(total_steps_m1 * end_pct) if end_pct is not None else total_steps
            adj_total_steps = end_step - adj_step
            schedule = self.schedule_configs[0].schedule if hasattr(self.schedule_configs[0], "schedule") else None
            if schedule is None:
                return 0.0
            return schedule.get_value(0, adj_total_steps)
        # adjust step/total_steps within SequentialSchedule to step/total_steps within schedule
        start_pct = config.start_percent
        assert start_pct is not None
        end_pct = config.end_percent
        start_step = int(total_steps_m1 * start_pct)
        adj_step = step - start_step
        end_step = total_steps if end_pct == 1.0 or end_pct is None else int(total_steps_m1 * end_pct)
        adj_total_steps = end_step - start_step
        if adj_total_steps == 0:
            return config.schedule.get_value(0, 1)
        if adj_step >= adj_total_steps:
            # return last value of previous schedule
            return config.schedule.get_value(adj_total_steps - 1, adj_total_steps)
        return config.schedule.get_value(adj_step, adj_total_steps)

    def __str__(self):
        return "\n".join(
            [
                type(self).__name__,
                "\n".join(f"  ({item[0]}): {item[1]}" for item in enumerate(self.schedule_configs)),
                ")",
            ]
        )


@dataclass
class SequentialStepScheduleConfig:
    schedule: ScheduleBase
    start_step: int | None = None
    end_step: int | None = None

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.start_step} - {self.end_step} {self.schedule}"


class SequentialStepSchedule(ScheduleBase):
    """A scheduler that switches between multiple schedules based on the step number."""

    def __init__(self, schedule_configs: Sequence[SequentialStepScheduleConfig], **kwargs):
        """Initialize the scheduler.

        Args:
            schedule_configs: A list of schedule configurations.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        if len(schedule_configs) == 0:
            raise ValueError("At least one schedule config must be provided.")
        self.schedule_configs = schedule_configs

        # if first schedule has no start_step -> set to 0 -> this ensures every schedule has a start_step
        if schedule_configs[0].start_step is None:
            schedule_configs[0].start_step = 0
        # propagate start/end
        for i in range(1, len(schedule_configs) - 1):
            # take start_step from previous schedule
            if schedule_configs[i].start_step is None:
                schedule_configs[i].start_step = schedule_configs[i - 1].end_step
            # take end_step from next schedule
            if schedule_configs[i].end_step is None:
                schedule_configs[i].end_step = schedule_configs[i + 1].start_step
        # edge case: last schedule propagate
        if len(schedule_configs) > 1:
            # propagate [-2].end_step forward
            if schedule_configs[-1].start_step is None:
                schedule_configs[-1].start_step = schedule_configs[-2].end_step
            # propagate [-1].start_step backward
            if schedule_configs[-2].end_step is None:
                schedule_configs[-2].end_step = schedule_configs[-1].start_step

        # check correctness of start/end
        if len(schedule_configs) == 1:
            # edge case: single schedule
            # always: 0 <= start
            # if end is not None: start <= end
            if not (0 <= schedule_configs[0].start_step):
                raise ValueError(
                    f"start_step must be >= 0, if a single schedule config is provided, got {schedule_configs[0].start_step}"
                )
            if schedule_configs[0].end_step is not None:
                if schedule_configs[0].start_step > schedule_configs[0].end_step:
                    raise ValueError(
                        f"start_step must be <= end_step, got {schedule_configs[0].start_step} > {schedule_configs[0].end_step}"
                    )
        else:
            # check 0 <= cfg[i].start <= cfg[i].end
            # check cfg[i].end <= cfg[i+1].start
            for i in range(len(schedule_configs) - 1):
                cur_config = schedule_configs[i]
                next_config = schedule_configs[i + 1]
                if cur_config.start_step is None or cur_config.end_step is None:
                    raise ValueError("start_step and end_step must be defined after propagation.")

                if not (0 <= cur_config.start_step):
                    raise ValueError(f"start_step must be >= 0, got {cur_config.start_step}")
                if cur_config.end_step is not None and not (cur_config.start_step <= cur_config.end_step):
                    raise ValueError(
                        f"start_step must be <= end_step, got {cur_config.start_step} > {cur_config.end_step}"
                    )
                if not (cur_config.end_step <= next_config.start_step):  # type: ignore[operator]
                    raise ValueError(
                        f"end_step must be <= next start_step, got {cur_config.end_step} > {next_config.start_step}"
                    )
            # last schedule is allowed to have no end_step
            last_config = schedule_configs[-1]
            if last_config.end_step is None:
                assert last_config.start_step is not None
                if not (0 <= last_config.start_step):
                    raise ValueError(f"start_step must be >= 0, got {last_config.start_step}")
            else:
                assert last_config.start_step is not None
                if not (0 <= last_config.start_step <= last_config.end_step):
                    raise ValueError(
                        f"start_step must be <= end_step, got {last_config.start_step} > {last_config.end_step}"
                    )

    def get_sequential_schedule_config(self, step: int) -> SequentialStepScheduleConfig | None:
        # step < config[0].start_step -> None
        # config[-1].end_step < step -> config[-1]
        for i in reversed(range(len(self.schedule_configs))):
            start_step = self.schedule_configs[i].start_step
            assert start_step is not None
            if start_step <= step:
                return self.schedule_configs[i]
        return None

    def _get_value(self, step: int, total_steps: int) -> float:
        config = self.get_sequential_schedule_config(step)
        if config is None:
            # adjust step/total_steps within SequentialSchedule to step/total_steps within schedule
            assert self.schedule_configs[0].start_step is not None
            adj_step = self.schedule_configs[0].start_step
            end_step = self.schedule_configs[0].end_step or total_steps
            adj_total_steps = end_step - adj_step
            return self.schedule_configs[0].schedule.get_value(0, adj_total_steps)
        # adjust step/total_steps within SequentialSchedule to step/total_steps within schedule
        assert config.start_step is not None
        adj_step = step - config.start_step
        end_step = config.end_step or total_steps
        adj_total_steps = end_step - config.start_step
        if adj_step >= adj_total_steps:
            # return last value of previous schedule
            return config.schedule.get_value(adj_total_steps - 1, adj_total_steps)
        return config.schedule.get_value(adj_step, adj_total_steps)

    def __str__(self):
        return "\n".join(
            [
                type(self).__name__,
                "\n".join(f"  ({item[0]}): {item[1]}" for item in enumerate(self.schedule_configs)),
                ")",
            ]
        )
