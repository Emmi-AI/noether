#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schedules.base import DecreasingProgressSchedule, ScheduleBase
from noether.core.schemas.schedules import (
    StepDecreasingScheduleConfig,
    StepFixedScheduleConfig,
    StepIntervalScheduleConfig,
)


class StepDecreasingSchedule(DecreasingProgressSchedule):
    """A scheduler that decreases exponentially from the maximum to minimum value over the total number of steps.

    Example:

        .. code-block:: yaml
            schedule_config:
                kind: noether.core.schedules.StepDecreasingSchedule
                factor: 0.1
                decreases_interval: 0.01
                max_value: ${model.optim.lr}

    I.e., after each 1% of the total training steps, the value is multiplied by 0.1.
    """

    def __init__(self, config: StepDecreasingScheduleConfig):
        """

        Args:
        config: The configuration for the scheduler. See :class:`noether.core.schemas.schedules.StepDecreasingScheduleConfig` for details.
        """
        super().__init__(config=config)
        self.factor = config.factor
        self.decreases_interval = config.decreases_interval

    def _get_progress(self, step: int, total_steps: int) -> float:
        progress = step / total_steps
        # round to 10th decimal place to avoid floating point precision errors
        step_idx = int(round(progress / self.decreases_interval, 10))
        return 1 - self.factor**step_idx


class StepFixedSchedule(ScheduleBase):
    """A scheduler that progresses at fixed steps and increases or decreases by some factor at these steps."""

    def __init__(self, config: StepFixedScheduleConfig):
        """

        Args:
            config: Configuration for the step fixed schedule.

        Example:
        .. code-block:: yaml
            schedule_config:
                kind: noether.core.schedules.StepFixedSchedule
                factor: 0.1
                start_value: ${model.optim.lr}
                steps:
                    - 0.01
                    - 0.02
                    - 0.03
        Lower LR by factor 0.1 at 1%, 2%, and 3% of total training steps.
        """
        super().__init__(overhang_percent=config.overhang_percent, overhang_steps=config.overhang_steps)
        self.steps = sorted(config.steps)
        self.start_value = config.start_value
        self.factor = config.factor

    def __str__(self):
        return f"{type(self).__name__}(start_value={self.start_value}, factor={self.factor}, steps={self.steps})"

    def _get_value(self, step: int, total_steps: int) -> float:
        progress = step / total_steps
        # search for step
        for i in range(len(self.steps)):
            if self.steps[i] > progress:
                step_idx = i
                break
        else:
            step_idx = len(self.steps)
        return self.start_value * self.factor**step_idx


class StepIntervalSchedule(ScheduleBase):
    """A scheduler that progresses at fixed intervals and increases or decreases by some factor at these intervals."""

    def __init__(self, config: StepIntervalScheduleConfig):
        """

        Args:
            config: Configuration for the step interval schedule.
        Example:

        .. code-block:: yaml
            schedule_config:
                kind: noether.core.schedules.StepIntervalSchedule
                start_value: 1.0
                factor: 0.5
                update_interval: 0.01
        """

        super().__init__(overhang_percent=config.overhang_percent, overhang_steps=config.overhang_steps)
        self.start_value = config.start_value
        self.factor = config.factor
        self.update_interval = config.update_interval

    def __str__(self):
        return (
            f"{type(self).__name__}"
            f"(start_value={self.start_value}, factor={self.factor}, interval={self.update_interval})"
        )

    def _get_value(self, step: int, total_steps: int) -> float:
        progress = step / total_steps
        # round to 10th decimal place to avoid floating point precision errors
        step_idx = int(round(progress / self.update_interval, 10))
        return self.start_value * self.factor**step_idx
