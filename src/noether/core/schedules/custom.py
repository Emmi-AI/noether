#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schedules.base import ScheduleBase
from noether.core.schemas.schedules import CustomScheduleConfig


class CustomSchedule(ScheduleBase):
    """Custom schedule that simply returns the values provided in the constructor.

    Example:

        .. code-block:: yaml
            schedule_config:
                kind: noether.core.schedules.CustomSchedule
                values:
                - 1.0e-3
                - 5.0e-4
                - 1.0e-4
    """

    def __init__(self, config: CustomScheduleConfig):
        """

        Args:
            config: Configuration of the custom schedule. See :class:`noether.core.schemas.schedules.CustomScheduleConfig` for details.
        """
        super().__init__(overhang_percent=config.overhang_percent, overhang_steps=config.overhang_steps)
        self.values = config.values

    def __str__(self):
        return f"{type(self).__name__}"

    def _get_value(self, step: int, total_steps: int) -> float:
        return self.values[step]
