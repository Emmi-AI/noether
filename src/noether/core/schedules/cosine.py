#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schedules.base import DecreasingProgressSchedule, IncreasingProgressSchedule
from noether.core.schedules.functional import cosine


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
