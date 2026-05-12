#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from typing import Union

from noether.core.schedules.constant import ConstantScheduleConfig
from noether.core.schedules.cosine import CosineDecreasingScheduleConfig, CosineIncreasingScheduleConfig
from noether.core.schedules.custom import CustomScheduleConfig
from noether.core.schedules.linear import LinearDecreasingScheduleConfig, LinearIncreasingScheduleConfig
from noether.core.schedules.linear_warmup_cosine_decay import LinearWarmupCosineDecayScheduleConfig
from noether.core.schedules.polynomial import PolynomialDecreasingScheduleConfig, PolynomialIncreasingScheduleConfig
from noether.core.schedules.schemas import (
    DecreasingProgressScheduleConfig,
    IncreasingProgressScheduleConfig,
    ProgressScheduleConfig,
    SchedulerConfig,
)
from noether.core.schedules.step import (
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
