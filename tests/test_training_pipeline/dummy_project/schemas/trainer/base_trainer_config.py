#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Union

from dummy_project.schemas.callbacks.base_callback_config import BoilerplateCallbackConfig
from pydantic import Field

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
from noether.training.trainers.base import BaseTrainerConfig

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

AllCallbacks = Union[BoilerplateCallbackConfig | CallbacksConfig]  #


class BaseTrainerConfig(BaseTrainerConfig):
    input_dim: int
    callbacks: list[AllCallbacks] | None = Field(..., description="List of callback configurations")
