#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Union

from noether.core.schedules.constant import ConstantScheduleConfig
from noether.core.schedules.custom import CustomScheduleConfig
from noether.core.schedules.linear_warmup_cosine_decay import LinearWarmupCosineDecayScheduleConfig
from noether.core.schedules.schemas import (
    CosineDecreasingScheduleConfig,
    CosineIncreasingScheduleConfig,
    DecreasingProgressScheduleConfig,
    IncreasingProgressScheduleConfig,
    LinearDecreasingScheduleConfig,
    LinearIncreasingScheduleConfig,
    PeriodicBoolScheduleConfig,
    PolynomialDecreasingScheduleConfig,
    PolynomialIncreasingScheduleConfig,
    ProgressScheduleConfig,
    ScheduleBaseConfig,
    SchedulerConfig,
    StepDecreasingScheduleConfig,
    StepFixedScheduleConfig,
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
    ScheduleBaseConfig,
]
