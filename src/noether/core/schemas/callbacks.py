#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for ``noether.core.schemas.callbacks``.

The callback config classes have moved next to the classes they configure in
:mod:`noether.core.callbacks` (and :mod:`noether.training.callbacks` for the
training-only callbacks). This module preserves the old import paths.
"""

from typing import Union

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
