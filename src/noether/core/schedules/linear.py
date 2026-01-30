#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schedules.base import DecreasingProgressSchedule, IncreasingProgressSchedule
from noether.core.schedules.functional import linear


class LinearDecreasingSchedule(DecreasingProgressSchedule):
    """A scheduler that decreases linearly from the maximum to minimum value over the total number of steps.

    Example:

        .. code-block:: yaml
            schedule_config:
            kind: noether.core.schedules.LinearDecreasingSchedule
            max_value: ${model.optim.lr}
            end_value: 0.0
    """

    def _get_progress(self, step: int, total_steps: int) -> float:
        return linear(step, total_steps)


class LinearIncreasingSchedule(IncreasingProgressSchedule):
    """A scheduler that increases linearly from the minimum to maximum value over the total number of steps.

    Example:
        >>> schedule_config:
        >>>   kind: noether.core.schedules.LinearIncreasingSchedule
        >>>   max_value: ${model.optim.lr}
        >>>   start_value: 0.0
    """

    def _get_progress(self, step: int, total_steps: int) -> float:
        return linear(step, total_steps)
