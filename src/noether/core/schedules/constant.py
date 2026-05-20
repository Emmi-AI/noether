#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from noether.core.schedules.base import ScheduleBase
from noether.core.schedules.schemas import ScheduleBaseConfig


class ConstantScheduleConfig(ScheduleBaseConfig):
    kind: Literal["noether.core.schedules.ConstantSchedule"] = "noether.core.schedules.ConstantSchedule"
    value: float
    """The constant value that will be returned for all steps. Value should be equal to the learning rate defined in the optimizer."""


class ConstantSchedule(ScheduleBase):
    """Constant value schedule that returns the same value for all steps.

    Example:

        .. code-block:: yaml

            schedule_config:
                kind: noether.core.schedules.ConstantSchedule
                value: ${model.optim.lr}
    """

    def __init__(self, config: ConstantScheduleConfig):
        """

        Args:
            config: Configuration of the constant schedule. See
                :class:`~noether.core.schemas.schedules.ConstantScheduleConfig` for details.
        """
        super().__init__(overhang_percent=config.overhang_percent, overhang_steps=config.overhang_steps)
        self.value = config.value

    def __str__(self):
        return f"{type(self).__name__}(value={self.value})"

    def _get_value(self, step: int, total_steps: int) -> float:
        return self.value
