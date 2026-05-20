#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from noether.core.schedules.base import (
    DecreasingProgressSchedule,
    DecreasingProgressScheduleConfig,
    IncreasingProgressSchedule,
    IncreasingProgressScheduleConfig,
)
from noether.core.schedules.functional import linear


class LinearDecreasingScheduleConfig(DecreasingProgressScheduleConfig):
    kind: Literal["noether.core.schedules.LinearDecreasingSchedule"] = "noether.core.schedules.LinearDecreasingSchedule"  # type: ignore[assignment]


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


class LinearIncreasingScheduleConfig(IncreasingProgressScheduleConfig):
    kind: Literal["noether.core.schedules.LinearIncreasingSchedule"] = "noether.core.schedules.LinearIncreasingSchedule"  # type: ignore[assignment]


class LinearIncreasingSchedule(IncreasingProgressSchedule):
    """A scheduler that increases linearly from the minimum to maximum value over the total number of steps.

    Example:

        .. code-block:: yaml

            schedule_config:
                kind: noether.core.schedules.LinearIncreasingSchedule
                max_value: ${model.optim.lr}
                start_value: 0.0
    """

    def _get_progress(self, step: int, total_steps: int) -> float:
        return linear(step, total_steps)
