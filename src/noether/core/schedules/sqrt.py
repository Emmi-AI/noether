#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schedules.base import DecreasingProgressSchedule
from noether.core.schedules.functional import sqrt_decay


class SqrtDecreasingSchedule(DecreasingProgressSchedule):
    """1-sqrt decay schedule (Hägele et al. 2024).

    Drops fast initially then tails off slowly: ``value(s) = max - (max - end) * sqrt(progress)``.
    Used as the cooldown phase of the Warmup-Stable-Decay schedule.

    Example:

        .. code-block:: yaml

            schedule_config:
                kind: noether.core.schedules.SqrtDecreasingSchedule
                max_value: ${model.optim.lr}
                end_value: 0.0
    """

    def _get_progress(self, step: int, total_steps: int) -> float:
        return sqrt_decay(step, total_steps)
