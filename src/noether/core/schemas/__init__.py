#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for ``noether.core.schemas``.

Schemas have been moved next to the classes they configure. The names below
keep the old ``from noether.core.schemas import X`` import paths alive, but
they are resolved lazily via :pep:`562` so that importing this package does
not eagerly load the new-home modules (which previously caused circular
imports).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # callbacks — point at canonical sources, never at the back-compat shim
    "BestCheckpointCallbackConfig": (
        "noether.core.callbacks.checkpoint.best_checkpoint",
        "BestCheckpointCallbackConfig",
    ),
    "BestMetricCallbackConfig": ("noether.core.callbacks.online.best_metric", "BestMetricCallbackConfig"),
    "CallBackBaseConfig": ("noether.core.callbacks.base", "CallBackBaseConfig"),
    "CallbacksConfig": ("noether.core.schemas.callbacks", "CallbacksConfig"),
    "CheckpointCallbackConfig": ("noether.core.callbacks.checkpoint.checkpoint", "CheckpointCallbackConfig"),
    "EmaCallbackConfig": ("noether.core.callbacks.checkpoint.ema", "EmaCallbackConfig"),
    "FixedEarlyStopperConfig": ("noether.core.callbacks.early_stoppers.fixed", "FixedEarlyStopperConfig"),
    "MetricEarlyStopperConfig": ("noether.core.callbacks.early_stoppers.metric", "MetricEarlyStopperConfig"),
    "OfflineLossCallbackConfig": ("noether.training.callbacks.offline_loss", "OfflineLossCallbackConfig"),
    "OnlineLossCallbackConfig": ("noether.core.callbacks.default.online_loss", "OnlineLossCallbackConfig"),
    "PeriodicDataIteratorCallbackConfig": (
        "noether.core.callbacks.periodic",
        "PeriodicDataIteratorCallbackConfig",
    ),
    "TrackAdditionalOutputsCallbackConfig": (
        "noether.core.callbacks.online.track_outputs",
        "TrackAdditionalOutputsCallbackConfig",
    ),
    # dataset — point at canonical sources, never at the back-compat shim
    "DatasetBaseConfig": ("noether.data.base.dataset", "DatasetBaseConfig"),
    "StandardDatasetConfig": ("noether.data.base.dataset", "StandardDatasetConfig"),
    # initializers — point at canonical sources, never at the back-compat shim
    "AnyInitializer": ("noether.core.initializers", "AnyInitializer"),
    "CheckpointInitializerConfig": ("noether.core.initializers", "CheckpointInitializerConfig"),
    "InitializerConfig": ("noether.core.initializers", "InitializerConfig"),
    "PreviousRunInitializerConfig": ("noether.core.initializers", "PreviousRunInitializerConfig"),
    "ResumeInitializerConfig": ("noether.core.initializers", "ResumeInitializerConfig"),
    # models — point at canonical sources, never at the back-compat shim package
    "ModelBaseConfig": ("noether.core.models.base", "ModelBaseConfig"),
    # normalizers
    "AnyNormalizer": ("noether.core.schemas.normalizers", "AnyNormalizer"),
    "FieldNormalizerConfig": ("noether.core.schemas.normalizers", "FieldNormalizerConfig"),
    # optimizers
    "AdamOptimizerConfig": ("noether.core.schemas.optimizers", "AdamOptimizerConfig"),
    "AnyOptimizerConfig": ("noether.core.schemas.optimizers", "AnyOptimizerConfig"),
    "MuonOptimizerConfig": ("noether.core.schemas.optimizers", "MuonOptimizerConfig"),
    "OptimizerConfig": ("noether.core.schemas.optimizers", "OptimizerConfig"),
    "ParamGroupModifierConfig": ("noether.core.schemas.optimizers", "ParamGroupModifierConfig"),
    "SGDOptimizerConfig": ("noether.core.schemas.optimizers", "SGDOptimizerConfig"),
    # schedules
    "AnyScheduleConfig": ("noether.core.schemas.schedules", "AnyScheduleConfig"),
    "ConstantScheduleConfig": ("noether.core.schemas.schedules", "ConstantScheduleConfig"),
    "CustomScheduleConfig": ("noether.core.schemas.schedules", "CustomScheduleConfig"),
    "DecreasingProgressScheduleConfig": ("noether.core.schemas.schedules", "DecreasingProgressScheduleConfig"),
    "IncreasingProgressScheduleConfig": ("noether.core.schemas.schedules", "IncreasingProgressScheduleConfig"),
    "LinearWarmupCosineDecayScheduleConfig": (
        "noether.core.schemas.schedules",
        "LinearWarmupCosineDecayScheduleConfig",
    ),
    "PeriodicBoolScheduleConfig": ("noether.core.schemas.schedules", "PeriodicBoolScheduleConfig"),
    "PolynomialDecreasingScheduleConfig": ("noether.core.schemas.schedules", "PolynomialDecreasingScheduleConfig"),
    "PolynomialIncreasingScheduleConfig": ("noether.core.schemas.schedules", "PolynomialIncreasingScheduleConfig"),
    "ProgressScheduleConfig": ("noether.core.schemas.schedules", "ProgressScheduleConfig"),
    "ScheduleBaseConfig": ("noether.core.schemas.schedules", "ScheduleBaseConfig"),
    "SchedulerConfig": ("noether.core.schemas.schedules", "SchedulerConfig"),
    "StepDecreasingScheduleConfig": ("noether.core.schemas.schedules", "StepDecreasingScheduleConfig"),
    "StepFixedScheduleConfig": ("noether.core.schemas.schedules", "StepFixedScheduleConfig"),
    "StepIntervalScheduleConfig": ("noether.core.schemas.schedules", "StepIntervalScheduleConfig"),
    # schema
    "ConfigSchema": ("noether.core.schemas.schema", "ConfigSchema"),
    # slurm
    "SlurmConfig": ("noether.core.schemas.slurm", "SlurmConfig"),
    # trackers
    "WandBTrackerSchema": ("noether.core.trackers", "WandBTrackerSchema"),
    # trainers
    "BaseTrainerConfig": ("noether.training.trainers.base", "BaseTrainerConfig"),
}

__all__ = [
    "AdamOptimizerConfig",
    "AnyInitializer",
    "AnyNormalizer",
    "AnyOptimizerConfig",
    "AnyScheduleConfig",
    "BaseTrainerConfig",
    "BestCheckpointCallbackConfig",
    "BestMetricCallbackConfig",
    "CallBackBaseConfig",
    "CallbacksConfig",
    "CheckpointCallbackConfig",
    "CheckpointInitializerConfig",
    "ConfigSchema",
    "ConstantScheduleConfig",
    "CustomScheduleConfig",
    "DatasetBaseConfig",
    "DecreasingProgressScheduleConfig",
    "EmaCallbackConfig",
    "FieldNormalizerConfig",
    "FixedEarlyStopperConfig",
    "IncreasingProgressScheduleConfig",
    "InitializerConfig",
    "LinearWarmupCosineDecayScheduleConfig",
    "MetricEarlyStopperConfig",
    "ModelBaseConfig",
    "MuonOptimizerConfig",
    "OfflineLossCallbackConfig",
    "OnlineLossCallbackConfig",
    "OptimizerConfig",
    "ParamGroupModifierConfig",
    "PeriodicBoolScheduleConfig",
    "PeriodicDataIteratorCallbackConfig",
    "PolynomialDecreasingScheduleConfig",
    "PolynomialIncreasingScheduleConfig",
    "PreviousRunInitializerConfig",
    "ProgressScheduleConfig",
    "ResumeInitializerConfig",
    "SGDOptimizerConfig",
    "ScheduleBaseConfig",
    "SchedulerConfig",
    "SlurmConfig",
    "StandardDatasetConfig",
    "StepDecreasingScheduleConfig",
    "StepFixedScheduleConfig",
    "StepIntervalScheduleConfig",
    "TrackAdditionalOutputsCallbackConfig",
    "WandBTrackerSchema",
]


def __getattr__(name: str) -> Any:
    try:
        module_path, attr = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(importlib.import_module(module_path), attr)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))


if TYPE_CHECKING:  # static type checkers — keep in sync with _LAZY_EXPORTS
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
    from noether.core.initializers import (
        AnyInitializer,
        CheckpointInitializerConfig,
        InitializerConfig,
        PreviousRunInitializerConfig,
        ResumeInitializerConfig,
    )
    from noether.core.models.base import ModelBaseConfig
    from noether.core.schedules.constant import ConstantScheduleConfig
    from noether.core.schedules.custom import CustomScheduleConfig
    from noether.core.schedules.linear_warmup_cosine_decay import LinearWarmupCosineDecayScheduleConfig
    from noether.core.schedules.polynomial import PolynomialDecreasingScheduleConfig, PolynomialIncreasingScheduleConfig
    from noether.core.schedules.schemas import (
        DecreasingProgressScheduleConfig,
        IncreasingProgressScheduleConfig,
        ProgressScheduleConfig,
        ScheduleBaseConfig,
        SchedulerConfig,
    )
    from noether.core.schedules.step import (
        StepDecreasingScheduleConfig,
        StepFixedScheduleConfig,
        StepIntervalScheduleConfig,
    )
    from noether.core.schemas.callbacks import CallbacksConfig
    from noether.core.schemas.normalizers import AnyNormalizer, FieldNormalizerConfig
    from noether.core.schemas.schema import ConfigSchema
    from noether.core.trackers import WandBTrackerSchema
    from noether.data.base.dataset import DatasetBaseConfig, StandardDatasetConfig
    from noether.training.callbacks.offline_loss import OfflineLossCallbackConfig
    from noether.training.cli.submit_job import SlurmConfig
    from noether.training.trainers.base import BaseTrainerConfig
