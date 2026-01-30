#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schedules.base import DecreasingProgressSchedule, IncreasingProgressSchedule
from noether.core.schedules.functional import polynomial
from noether.core.schemas.schedules import PolynomialDecreasingScheduleConfig, PolynomialIncreasingScheduleConfig


class PolynomialDecreasingSchedule(DecreasingProgressSchedule):
    """A scheduler that decreases polynomially from the maximum to minimum value over the total number of steps."""

    def __init__(self, config: PolynomialDecreasingScheduleConfig):
        """

        Args:
            config: Configuration for the polynomial decreasing schedule. See :class:`noether.core.schemas.schedules.PolynomialDecreasingScheduleConfig` for details.

        Example:
            .. code-block:: yaml
                schedule_config:
                    kind: noether.core.schedules.PolynomialDecreasingSchedule
                    power: 2.0
                    start_value: ${model.optim.lr} # reference to the lr defined above
                    end_value: 1e-6
        """
        super().__init__(config=config)
        self.power = config.power

    def _get_progress(self, step: int, total_steps: int) -> float:
        return polynomial(step, total_steps, power=self.power)


class PolynomialIncreasingSchedule(IncreasingProgressSchedule):
    """A scheduler that increases polynomially from the minimum to maximum value over the total number of steps."""

    def __init__(self, config: PolynomialIncreasingScheduleConfig):
        """

        Args:
            config: Configuration for the polynomial increasing schedule. See :class:`noether.core.schemas.schedules.PolynomialIncreasingScheduleConfig` for details.

        Example:
            .. code-block:: yaml
                 schedule_config:
                     kind: noether.core.schedules.PolynomialIncreasingSchedule
                     power: 2.0
                     start_value: 1e-6
                     max_value: ${model.optim.lr} # reference to the lr defined above
        """
        super().__init__(config=config)
        self.power = config.power

    def _get_progress(self, step: int, total_steps: int) -> float:
        return polynomial(step, total_steps, power=self.power)
