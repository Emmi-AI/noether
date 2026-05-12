#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from noether.core.schedules.base import (
    DecreasingProgressSchedule,
    DecreasingProgressScheduleConfig,
    IncreasingProgressSchedule,
)
from noether.core.schedules.functional import cosine
from noether.core.schedules.schemas import IncreasingProgressScheduleConfig


class CosineDecreasingScheduleConfig(DecreasingProgressScheduleConfig):
    kind: Literal["noether.core.schedules.CosineDecreasingSchedule"] = "noether.core.schedules.CosineDecreasingSchedule"  # type: ignore[assignment]


class CosineDecreasingSchedule(DecreasingProgressSchedule):
    """Cosine annealing scheduler with decreasing values.

    Example:

        .. code-block:: yaml

            schedule_config:
                kind: noether.core.schedules.CosineDecreasingSchedule
                max_value: ${model.optim.lr} # or just manually set the starting value
                end_value: 0.0
    """

    def _get_progress(self, step: int, total_steps: int) -> float:
        return cosine(step, total_steps)


class CosineIncreasingScheduleConfig(IncreasingProgressScheduleConfig):
    kind: Literal["noether.core.schedules.CosineIncreasingSchedule"] = "noether.core.schedules.CosineIncreasingSchedule"  # type: ignore[assignment]


class CosineIncreasingSchedule(IncreasingProgressSchedule):
    """Cosine annealing scheduler with increasing values.

    Example:

        .. code-block:: yaml

            schedule_config:
                kind: noether.core.schedules.CosineIncreasingSchedule
                max_value: ${model.optim.lr}
                start_value: 0.0
    """

    def _get_progress(self, step: int, total_steps: int) -> float:
        return cosine(step, total_steps)
