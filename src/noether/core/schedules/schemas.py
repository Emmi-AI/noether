#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ScheduleBaseConfig(BaseModel):
    kind: str | None = Field("noether.core.schedules.base.ScheduleBase")
    """The fully qualified class name of the scheduler."""
    overhang_percent: float | None = Field(None)
    """The percentage by which the schedule is artificially prolonged. Mutually exclusive with `overhang_steps`."""
    overhang_steps: int | None = Field(None)
    """The number of steps by which the schedule is artificially prolonged. Mutually exclusive with `overhang_percent`."""

    start_value: float = Field(0.0, ge=0.0)

    end_value: float = Field(1e-6, ge=0.0)

    weight_decay: float | None = Field(0.0)

    start_percent: float | None = Field(None, ge=0.0, le=1.0)
    """The percentage of steps at which the schedule starts."""
    end_percent: float | None = Field(None, ge=0.0, le=1.0)
    """The percentage of steps at which the schedule ends."""

    start_step: int | None = Field(None, ge=0)
    """The step at which the schedule starts."""
    end_step: int | None = Field(None, ge=0)
    """The step at which the schedule ends."""

    interval: Literal["update", "epoch"] = Field("update")
    """Whether the schedule is based on updates or epochs. Interval should be either "update" or "epoch". Default is "update". Under the hood steps is always used. However, when "epoch" is selected here, the step count is derived from epochs via the UpdateCounter."""

    @model_validator(mode="after")
    def check_mutual_exclusion(self) -> "ScheduleBaseConfig":
        """
        Ensures that 'overhang_percent' and 'overhang_steps' are mutually exclusive.
        """

        if self.overhang_percent is not None and self.overhang_steps is not None:
            raise ValueError("overhang_percent and overhang_steps are mutually exclusive")
        return self

    @model_validator(mode="after")
    def validate_start_end_steps(self) -> "ScheduleBaseConfig":
        if not type(self.start_step) == type(self.end_step):
            raise ValueError("start_step and end_step must both be defined or both be None")
        if self.start_step and self.end_step:
            if self.start_percent is not None or self.end_percent is not None:
                raise ValueError("Cannot define both start_step/end_step and start_percent/end_percent")
            if self.start_step >= self.end_step:
                raise ValueError("start_step must be less than end_step")
            else:
                return self
        else:
            return self

    @model_validator(mode="after")
    def validate_start_end_percents(self) -> "ScheduleBaseConfig":
        if not type(self.start_percent) == type(self.end_percent):
            raise ValueError("start_percent and end_percent must both be defined or both be None")
        if self.start_percent and self.end_percent:
            if self.start_step is not None or self.end_step is not None:
                raise ValueError("Cannot define both start_step/end_step and start_percent/end_percent")
            if self.start_percent >= self.end_percent:
                raise ValueError("start_percent must be less than end_percent")
            else:
                return self
        else:
            return self


class ProgressScheduleConfig(ScheduleBaseConfig):
    kind: Literal["noether.core.schedules.ProgressSchedule"] = "noether.core.schedules.ProgressSchedule"

    exclude_first: bool = Field(False)
    """Whether to exclude the first value of the schedule."""
    exclude_last: bool = Field(False)
    """Whether to exclude the last value of the schedule."""


class SchedulerConfig(ScheduleBaseConfig):
    kind: Literal["noether.core.schedules.scheduler.SchedulerConfig"] = (
        "noether.core.schedules.scheduler.SchedulerConfig"
    )
    warmup_percent: float = Field(..., ge=0.0, le=1.0)
    end_value: float = Field(..., ge=0.0)


class DecreasingProgressScheduleConfig(ProgressScheduleConfig):
    kind: Literal["noether.core.schedules.DecreasingProgressSchedule"] = (
        "noether.core.schedules.DecreasingProgressSchedule"  # type: ignore[assignment]
    )
    max_value: float = Field(..., ge=0.0)
    """Maximum (starting) value of the schedule."""
    end_value: float = Field(0.0, ge=0.0)
    """Minimum (ending) value of the schedule."""


class IncreasingProgressScheduleConfig(ProgressScheduleConfig):
    kind: Literal["noether.core.schedules.IncreasingProgressSchedule"] = (
        "noether.core.schedules.IncreasingProgressSchedule"  # type: ignore[assignment]
    )
    start_value: float = Field(0.0)
    max_value: float | None = Field(...)
    """Minimum (starting) value of the schedule."""
