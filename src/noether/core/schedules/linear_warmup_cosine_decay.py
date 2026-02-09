#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schedules.base import (
    ScheduleBase,
    SequentialPercentSchedule,
    SequentialPercentScheduleConfig,
    SequentialStepSchedule,
    SequentialStepScheduleConfig,
)
from noether.core.schedules.cosine import CosineDecreasingSchedule
from noether.core.schedules.linear import LinearIncreasingSchedule
from noether.core.schemas.schedules import (
    DecreasingProgressScheduleConfig,
    IncreasingProgressScheduleConfig,
    LinearWarmupCosineDecayScheduleConfig,
)


class LinearWarmupCosineDecaySchedule(ScheduleBase):
    """A cosine annealing scheduler with linear increasing warmup phase."

    Example:

        .. code-block:: yaml
            schedule_config:
            kind: noether.core.schedules.LinearWarmupCosineDecaySchedule
            warmup_percent: 0.05
            end_value: 1.0e-6
            max_value: ${model.optim.lr} # or just manually set the max value
    """

    schedule: ScheduleBase

    def __init__(
        self,
        config: LinearWarmupCosineDecayScheduleConfig,
    ):
        """

        Takes either warmup_steps or warmup_percent as argument to determine the length of the warmup phase.

        Args:
            config: Configuration for the linear warmup cosine decay schedule. See
                :class:`~noether.core.schemas.schedules.LinearWarmupCosineDecayScheduleConfig` for details.
        """
        super().__init__(overhang_percent=config.overhang_percent, overhang_steps=config.overhang_steps)
        self.warmup_steps = config.warmup_steps
        self.warmup_percent = config.warmup_percent

        if self.warmup_steps is not None and self.warmup_percent is not None:
            raise ValueError("Only one of warmup_steps or warmup_percent can be set.")

        if self.warmup_steps is not None:
            self.schedule = SequentialStepSchedule(
                schedule_configs=[
                    SequentialStepScheduleConfig(
                        schedule=LinearIncreasingSchedule(
                            config=IncreasingProgressScheduleConfig.model_validate(
                                dict(
                                    exclude_first=config.start_value == 0,
                                    exclude_last=True,
                                    start_value=config.start_value,
                                    max_value=config.max_value,
                                )
                            )
                        ),
                        start_step=0,
                        end_step=self.warmup_steps,
                    ),
                    SequentialStepScheduleConfig(
                        schedule=CosineDecreasingSchedule(
                            config=DecreasingProgressScheduleConfig.model_validate(
                                dict(
                                    exclude_first=False,
                                    exclude_last=False,
                                    max_value=config.max_value,
                                    end_value=config.end_value,
                                )
                            )
                        ),
                        start_step=self.warmup_steps,
                    ),
                ],
            )
        elif self.warmup_percent is not None:
            self.schedule = SequentialPercentSchedule(
                schedule_configs=[
                    SequentialPercentScheduleConfig(
                        schedule=LinearIncreasingSchedule(
                            config=IncreasingProgressScheduleConfig.model_validate(
                                dict(
                                    exclude_first=config.start_value == 0,
                                    exclude_last=True,
                                    start_value=config.start_value,
                                    max_value=config.max_value,
                                )
                            )
                        ),
                        end_percent=self.warmup_percent,
                    ),
                    SequentialPercentScheduleConfig(
                        schedule=CosineDecreasingSchedule(
                            config=DecreasingProgressScheduleConfig.model_validate(
                                dict(
                                    exclude_first=False,
                                    exclude_last=False,
                                    max_value=config.max_value,
                                    end_value=config.end_value,
                                )
                            )
                        ),
                        start_percent=self.warmup_percent,
                    ),
                ],
            )

    def _get_value(self, step: int, total_steps: int) -> float:
        return self.schedule.get_value(step=step, total_steps=total_steps)

    def __str__(self):
        if self.warmup_percent is not None:
            return f"{type(self).__name__}(warmup_percent={self.warmup_percent})"
        if self.warmup_steps is not None:
            return f"{type(self).__name__}(warmup_steps={self.warmup_steps})"
        raise RuntimeError
