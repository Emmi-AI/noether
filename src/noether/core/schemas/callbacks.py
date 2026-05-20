#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for callback configs.

The callback config classes have moved next to the classes they configure in
:mod:`noether.core.callbacks` (and :mod:`noether.training.callbacks` for the
training-only callbacks). ``CallbacksConfig`` is the discriminated union of
all callback configs; it is defined here for back-compat only.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from noether.core.callbacks.base import CallBackBaseConfig
    from noether.core.callbacks.checkpoint.best_checkpoint import BestCheckpointCallbackConfig
    from noether.core.callbacks.checkpoint.checkpoint import CheckpointCallbackConfig
    from noether.core.callbacks.checkpoint.ema import EmaCallbackConfig
    from noether.core.callbacks.default.online_loss import OnlineLossCallbackConfig
    from noether.core.callbacks.early_stoppers.fixed import FixedEarlyStopperConfig
    from noether.core.callbacks.early_stoppers.metric import MetricEarlyStopperConfig
    from noether.core.callbacks.online.best_metric import BestMetricCallbackConfig
    from noether.core.callbacks.online.track_outputs import TrackAdditionalOutputsCallbackConfig
    from noether.core.callbacks.periodic import PeriodicDataIteratorCallbackConfig
    from noether.training.callbacks.offline_loss import OfflineLossCallbackConfig
    from noether.training.callbacks.profiler import PyTorchProfilerCallbackConfig

    CallbacksConfig = Union[
        BestCheckpointCallbackConfig
        | CheckpointCallbackConfig
        | EmaCallbackConfig
        | OnlineLossCallbackConfig
        | BestMetricCallbackConfig
        | TrackAdditionalOutputsCallbackConfig
        | OfflineLossCallbackConfig
        | MetricEarlyStopperConfig
        | FixedEarlyStopperConfig
        | PeriodicDataIteratorCallbackConfig
        | PyTorchProfilerCallbackConfig
    ]

__all__ = [
    "BestCheckpointCallbackConfig",
    "BestMetricCallbackConfig",
    "CallBackBaseConfig",
    "CallbacksConfig",
    "CheckpointCallbackConfig",
    "EmaCallbackConfig",
    "FixedEarlyStopperConfig",
    "MetricEarlyStopperConfig",
    "OfflineLossCallbackConfig",
    "OnlineLossCallbackConfig",
    "PeriodicDataIteratorCallbackConfig",
    "PyTorchProfilerCallbackConfig",
    "TrackAdditionalOutputsCallbackConfig",
]

_LAZY: dict[str, str] = {
    "BestCheckpointCallbackConfig": "noether.core.callbacks.checkpoint.best_checkpoint",
    "BestMetricCallbackConfig": "noether.core.callbacks.online.best_metric",
    "CallBackBaseConfig": "noether.core.callbacks.base",
    "CheckpointCallbackConfig": "noether.core.callbacks.checkpoint.checkpoint",
    "EmaCallbackConfig": "noether.core.callbacks.checkpoint.ema",
    "FixedEarlyStopperConfig": "noether.core.callbacks.early_stoppers.fixed",
    "MetricEarlyStopperConfig": "noether.core.callbacks.early_stoppers.metric",
    "OfflineLossCallbackConfig": "noether.training.callbacks.offline_loss",
    "OnlineLossCallbackConfig": "noether.core.callbacks.default.online_loss",
    "PeriodicDataIteratorCallbackConfig": "noether.core.callbacks.periodic",
    "PyTorchProfilerCallbackConfig": "noether.training.callbacks.profiler",
    "TrackAdditionalOutputsCallbackConfig": "noether.core.callbacks.online.track_outputs",
}


def _build_callbacks_config() -> Any:
    from noether.core.callbacks.checkpoint.best_checkpoint import BestCheckpointCallbackConfig
    from noether.core.callbacks.checkpoint.checkpoint import CheckpointCallbackConfig
    from noether.core.callbacks.checkpoint.ema import EmaCallbackConfig
    from noether.core.callbacks.default.online_loss import OnlineLossCallbackConfig
    from noether.core.callbacks.early_stoppers.fixed import FixedEarlyStopperConfig
    from noether.core.callbacks.early_stoppers.metric import MetricEarlyStopperConfig
    from noether.core.callbacks.online.best_metric import BestMetricCallbackConfig
    from noether.core.callbacks.online.track_outputs import TrackAdditionalOutputsCallbackConfig
    from noether.core.callbacks.periodic import PeriodicDataIteratorCallbackConfig
    from noether.training.callbacks.offline_loss import OfflineLossCallbackConfig
    from noether.training.callbacks.profiler import PyTorchProfilerCallbackConfig

    return Union[
        BestCheckpointCallbackConfig
        | CheckpointCallbackConfig
        | EmaCallbackConfig
        | OnlineLossCallbackConfig
        | BestMetricCallbackConfig
        | TrackAdditionalOutputsCallbackConfig
        | OfflineLossCallbackConfig
        | MetricEarlyStopperConfig
        | FixedEarlyStopperConfig
        | PeriodicDataIteratorCallbackConfig
        | PyTorchProfilerCallbackConfig
    ]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_path = _LAZY[name]
        warnings.warn(
            f"Importing `{name}` from `{__name__}` is deprecated; import from `{module_path}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(module_path), name)
    if name == "CallbacksConfig":
        warnings.warn(
            "Importing `CallbacksConfig` from `noether.core.schemas.callbacks` is deprecated; "
            "build the union from the individual callback configs in `noether.core.callbacks` "
            "(and `noether.training.callbacks`) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _build_callbacks_config()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
