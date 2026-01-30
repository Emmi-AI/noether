#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


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


class ConstantScheduleConfig(ScheduleBaseConfig):
    kind: Literal["noether.core.schedules.ConstantSchedule"] = "noether.core.schedules.ConstantSchedule"
    value: float
    """The constant value that will be returned for all steps. Value should be equal to the learning rate defined in the optimizer."""


class CustomScheduleConfig(ScheduleBaseConfig):
    kind: Literal["noether.core.schedules.CustomSchedule"] = "noether.core.schedules.CustomSchedule"
    values: list[float]
    """The list of values that will be returned for each step. Values show ben as long as the number of steps."""


class LinearWarmupCosineDecayScheduleConfig(ScheduleBaseConfig):
    kind: Literal["noether.core.schedules.LinearWarmupCosineDecaySchedule"] = (
        "noether.core.schedules.LinearWarmupCosineDecaySchedule"
    )
    warmup_steps: int | None = None
    """The number of steps to linearly increase the value from start to max."""
    warmup_percent: float | None = None
    """The percentage of steps to linearly increase the value from start to max."""
    max_value: float = Field(1.0)
    """The maximum value of the scheduler from which to start the cosine decay phase. This should be equal to the learning rate defined in the optimizer. I.e., max value is learning rate"""

    @model_validator(mode="after")
    def validate_warmup(self) -> "LinearWarmupCosineDecayScheduleConfig":
        """
        Ensures that exactly one of 'warmup_steps' or 'warmup_percent' is specified.
        """
        if (self.warmup_steps is None) == (self.warmup_percent is None):
            raise ValueError("Define exactly one of warmup_steps or warmup_percent")
        return self


class LinearIncreasingScheduleConfig(IncreasingProgressScheduleConfig):
    kind: Literal["noether.core.schedules.LinearIncreasingSchedule"] = "noether.core.schedules.LinearIncreasingSchedule"  # type: ignore[assignment]


class LinearDecreasingScheduleConfig(DecreasingProgressScheduleConfig):
    kind: Literal["noether.core.schedules.LinearDecreasingSchedule"] = "noether.core.schedules.LinearDecreasingSchedule"  # type: ignore[assignment]


class PeriodicBoolScheduleConfig(ScheduleBaseConfig):
    kind: Literal["noether.core.schedules.PeriodicBoolSchedule"] = "noether.core.schedules.PeriodicBoolSchedule"
    initial_state: bool
    """The initial (boolean) state of the scheduler (on or off)."""
    off_value: float = Field(0.0)
    """The value to return when the scheduler is in the off state."""
    on_value: float = Field(1.0)
    """ The value to return when the scheduler is in the on state."""
    off_duration: int = Field(1)
    """The number of steps the scheduler is in the off state."""
    on_duration: int = Field(1)
    """The number of steps the scheduler is in the on state."""
    invert: bool = Field(False)
    """Whether to invert the scheduler, i.e. return off_value when on and vice versa."""


class PolynomialDecreasingScheduleConfig(DecreasingProgressScheduleConfig):
    kind: Literal["noether.core.schedules.PolynomialDecreasingSchedule"] = (
        "noether.core.schedules.PolynomialDecreasingSchedule"  # type: ignore[assignment]
    )
    power: float = Field(1.0)
    """The power of the polynomial function."""


class PolynomialIncreasingScheduleConfig(IncreasingProgressScheduleConfig):
    kind: Literal["noether.core.schedules.PolynomialIncreasingSchedule"] = (
        "noether.core.schedules.PolynomialIncreasingSchedule"  # type: ignore[assignment]
    )
    power: float = Field(1.0)
    """The power of the polynomial function."""


class StepDecreasingScheduleConfig(DecreasingProgressScheduleConfig):
    kind: Literal["noether.core.schedules.StepDecreasingSchedule"] = "noether.core.schedules.StepDecreasingSchedule"  # type: ignore[assignment]
    factor: float = Field(..., ge=0.0)
    """The factor by which the value decreases."""
    decreases_interval: float = Field(..., gt=0.0, lt=1.0)
    """The interval in range [0, 1] at which the value decreases."""
    # max_value: float = Field(None)

    @model_validator(mode="after")
    def check_interval(self) -> "StepDecreasingScheduleConfig":
        """
        Ensures that 'interval' is a float in the range (0, 1).
        """
        if not (isinstance(self.decreases_interval, int | float) and 0.0 < self.decreases_interval < 1.0):
            raise ValueError("interval must be a float in the range (0, 1)")
        return self


class StepFixedScheduleConfig(ScheduleBaseConfig):
    kind: Literal["noether.core.schedules.StepFixedSchedule"] = "noether.core.schedules.StepFixedSchedule"
    start_value: float = Field(1.0)
    """The initial value of the scheduler."""
    factor: float = Field(..., ge=0.0)
    """The factor by which the value is multiplied after reaching the next step provided in steps."""
    steps: list[float] = Field(...)
    """The steps at which the value changes, must be a list of floats in the range (0, 1)."""

    @model_validator(mode="after")
    def validate_steps(self) -> "StepFixedScheduleConfig":
        """
        Ensures that 'steps' is a non-empty list of floats in the range (0, 1).
        """
        if not (isinstance(self.steps, list) and len(self.steps) > 0):
            raise ValueError("steps must be a non-empty list")
        if not all(isinstance(step, int | float) and 0.0 < step < 1.0 for step in self.steps):
            raise ValueError("all steps must be floats in the range (0, 1)")
        return self


class StepIntervalScheduleConfig(ScheduleBaseConfig):
    kind: Literal["noether.core.schedules.StepIntervalSchedule"] = "noether.core.schedules.StepIntervalSchedule"
    start_value: float = Field(1.0)
    """The initial value of the scheduler. I.e, the learning rate at step 0."""
    factor: float = Field(..., ge=0.0)
    """The factor by which the value is multiplied after reaching the next interval."""
    update_interval: float = Field(..., gt=0.0, lt=1.0)
    """The interval in range (0, 1) at which the value changes."""

    @field_validator("update_interval")
    def check_update_interval(cls, v: float) -> float:
        """
        Ensures that 'update_interval' is a float in the range (0, 1).
        """
        if not (isinstance(v, int | float) and 0.0 < v < 1.0):
            raise ValueError("update_interval must be a float in the range (0, 1)")
        return v


class CosineDecreasingScheduleConfig(DecreasingProgressScheduleConfig):
    kind: Literal["noether.core.schedules.CosineDecreasingSchedule"] = "noether.core.schedules.CosineDecreasingSchedule"  # type: ignore[assignment]


class CosineIncreasingScheduleConfig(IncreasingProgressScheduleConfig):
    kind: Literal["noether.core.schedules.CosineIncreasingSchedule"] = "noether.core.schedules.CosineIncreasingSchedule"  # type: ignore[assignment]


AnyScheduleConfig = Union[
    SchedulerConfig,
    DecreasingProgressScheduleConfig,
    IncreasingProgressScheduleConfig,
    ProgressScheduleConfig,
    ConstantScheduleConfig,
    CustomScheduleConfig,
    LinearWarmupCosineDecayScheduleConfig,
    PeriodicBoolScheduleConfig,
    PolynomialDecreasingScheduleConfig,
    PolynomialIncreasingScheduleConfig,
    StepDecreasingScheduleConfig,
    StepFixedScheduleConfig,
    StepIntervalScheduleConfig,
    CosineDecreasingScheduleConfig,
    CosineIncreasingScheduleConfig,
    LinearIncreasingScheduleConfig,
    LinearDecreasingScheduleConfig,
]
