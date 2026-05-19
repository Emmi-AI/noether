#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Union

from .base import (
    DecreasingProgressSchedule,
    IncreasingProgressSchedule,
    ProgressSchedule,
    ScheduleBase,
    SequentialPercentSchedule,
    SequentialPercentScheduleConfig,
    SequentialStepSchedule,
    SequentialStepScheduleConfig,
)
from .constant import ConstantSchedule, ConstantScheduleConfig
from .cosine import (
    CosineDecreasingSchedule,
    CosineDecreasingScheduleConfig,
    CosineIncreasingSchedule,
    CosineIncreasingScheduleConfig,
)
from .custom import CustomSchedule, CustomScheduleConfig
from .functional import cosine, linear, polynomial
from .linear import (
    LinearDecreasingSchedule,
    LinearDecreasingScheduleConfig,
    LinearIncreasingSchedule,
    LinearIncreasingScheduleConfig,
)
from .linear_warmup_cosine_decay import LinearWarmupCosineDecaySchedule, LinearWarmupCosineDecayScheduleConfig
from .polynomial import (
    PolynomialDecreasingSchedule,
    PolynomialDecreasingScheduleConfig,
    PolynomialIncreasingSchedule,
    PolynomialIncreasingScheduleConfig,
)
from .schemas import (
    DecreasingProgressScheduleConfig,
    IncreasingProgressScheduleConfig,
    ProgressScheduleConfig,
    ScheduleBaseConfig,
    SchedulerConfig,
)
from .step import (
    StepDecreasingSchedule,
    StepDecreasingScheduleConfig,
    StepFixedSchedule,
    StepFixedScheduleConfig,
    StepIntervalSchedule,
    StepIntervalScheduleConfig,
)

AnyScheduleConfig = Union[
    SchedulerConfig,
    DecreasingProgressScheduleConfig,
    IncreasingProgressScheduleConfig,
    ProgressScheduleConfig,
    ConstantScheduleConfig,
    CustomScheduleConfig,
    LinearWarmupCosineDecayScheduleConfig,
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

__all__ = [
    # --- from base:
    "DecreasingProgressSchedule",
    "IncreasingProgressSchedule",
    "ProgressSchedule",
    "ScheduleBase",
    "SequentialStepSchedule",
    "SequentialStepScheduleConfig",
    "SequentialPercentSchedule",
    "SequentialPercentScheduleConfig",
    # --- from functional:
    "cosine",
    "linear",
    "polynomial",
    # --- schedules and their configs:
    "AnyScheduleConfig",
    "ConstantSchedule",
    "ConstantScheduleConfig",
    "CosineDecreasingSchedule",
    "CosineDecreasingScheduleConfig",
    "CosineIncreasingSchedule",
    "CosineIncreasingScheduleConfig",
    "CustomSchedule",
    "CustomScheduleConfig",
    "DecreasingProgressScheduleConfig",
    "IncreasingProgressScheduleConfig",
    "LinearDecreasingSchedule",
    "LinearDecreasingScheduleConfig",
    "LinearIncreasingSchedule",
    "LinearIncreasingScheduleConfig",
    "LinearWarmupCosineDecaySchedule",
    "LinearWarmupCosineDecayScheduleConfig",
    "PolynomialDecreasingSchedule",
    "PolynomialDecreasingScheduleConfig",
    "PolynomialIncreasingSchedule",
    "PolynomialIncreasingScheduleConfig",
    "ProgressScheduleConfig",
    "ScheduleBaseConfig",
    "SchedulerConfig",
    "StepDecreasingSchedule",
    "StepDecreasingScheduleConfig",
    "StepFixedSchedule",
    "StepFixedScheduleConfig",
    "StepIntervalSchedule",
    "StepIntervalScheduleConfig",
]
