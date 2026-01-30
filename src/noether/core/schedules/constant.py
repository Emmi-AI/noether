#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schedules.base import ScheduleBase
from noether.core.schemas.schedules import ConstantScheduleConfig


class ConstantSchedule(ScheduleBase):
    """Constant value schedule that returns the same value for all steps.

    Example:

        .. code-block:: yaml
            schedule_config:
            kind: noether.core.schedules.ConstantSchedule
            value : ${model.optim.lr}
    """

    def __init__(self, config: ConstantScheduleConfig):
        """

        Args:
            scheduler_config: Configuration of the constant schedule. See :class:`noether.core.schemas.schedules.ConstantScheduleConfig` for details.
        """
        super().__init__(overhang_percent=config.overhang_percent, overhang_steps=config.overhang_steps)
        self.value = config.value

    def __str__(self):
        return f"{type(self).__name__}(value={self.value})"

    def _get_value(self, step: int, total_steps: int) -> float:
        return self.value
