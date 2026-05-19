#  Copyright © 2026 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for schedule configs.

The canonical home is :mod:`noether.core.schedules`.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from noether.core.schedules import (
        AnyScheduleConfig,
        ConstantScheduleConfig,
        CosineDecreasingScheduleConfig,
        CosineIncreasingScheduleConfig,
        CustomScheduleConfig,
        DecreasingProgressScheduleConfig,
        IncreasingProgressScheduleConfig,
        LinearDecreasingScheduleConfig,
        LinearIncreasingScheduleConfig,
        LinearWarmupCosineDecayScheduleConfig,
        PolynomialDecreasingScheduleConfig,
        PolynomialIncreasingScheduleConfig,
        ProgressScheduleConfig,
        ScheduleBaseConfig,
        SchedulerConfig,
        StepDecreasingScheduleConfig,
        StepFixedScheduleConfig,
        StepIntervalScheduleConfig,
    )

__all__ = [
    "AnyScheduleConfig",
    "ConstantScheduleConfig",
    "CosineDecreasingScheduleConfig",
    "CosineIncreasingScheduleConfig",
    "CustomScheduleConfig",
    "DecreasingProgressScheduleConfig",
    "IncreasingProgressScheduleConfig",
    "LinearDecreasingScheduleConfig",
    "LinearIncreasingScheduleConfig",
    "LinearWarmupCosineDecayScheduleConfig",
    "PolynomialDecreasingScheduleConfig",
    "PolynomialIncreasingScheduleConfig",
    "ProgressScheduleConfig",
    "ScheduleBaseConfig",
    "SchedulerConfig",
    "StepDecreasingScheduleConfig",
    "StepFixedScheduleConfig",
    "StepIntervalScheduleConfig",
]

_LAZY: dict[str, str] = dict.fromkeys(__all__, "noether.core.schedules")


def __getattr__(name: str) -> Any:
    try:
        module_path = _LAZY[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    warnings.warn(
        f"Importing `{name}` from `{__name__}` is deprecated; import from `{module_path}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(module_path), name)
