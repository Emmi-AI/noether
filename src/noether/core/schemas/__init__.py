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
import warnings
from typing import TYPE_CHECKING, Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # callbacks — point at canonical sources, never at the back-compat shim
    "BestCheckpointCallbackConfig": (
        "noether.core.callbacks.checkpoint.best_checkpoint",
        "BestCheckpointCallbackConfig",
    ),
    "BestMetricCallbackConfig": ("noether.core.callbacks.online.best_metric", "BestMetricCallbackConfig"),
    "CallBackBaseConfig": ("noether.core.callbacks.base", "CallBackBaseConfig"),
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
    "AnyNormalizer": ("noether.data.preprocessors.normalizers", "AnyNormalizer"),
    "FieldNormalizerConfig": ("noether.data.preprocessors.normalizers", "FieldNormalizerConfig"),
    # optimizers
    "AdamOptimizerConfig": ("noether.core.optimizer.schemas", "AdamOptimizerConfig"),
    "AnyOptimizerConfig": ("noether.core.optimizer.schemas", "AnyOptimizerConfig"),
    "MuonOptimizerConfig": ("noether.core.optimizer.schemas", "MuonOptimizerConfig"),
    "OptimizerConfig": ("noether.core.optimizer.schemas", "OptimizerConfig"),
    "ParamGroupModifierConfig": ("noether.core.optimizer.schemas", "ParamGroupModifierConfig"),
    "SGDOptimizerConfig": ("noether.core.optimizer.schemas", "SGDOptimizerConfig"),
    # schedules
    "AnyScheduleConfig": ("noether.core.schedules", "AnyScheduleConfig"),
    "ConstantScheduleConfig": ("noether.core.schedules", "ConstantScheduleConfig"),
    "CustomScheduleConfig": ("noether.core.schedules", "CustomScheduleConfig"),
    "DecreasingProgressScheduleConfig": ("noether.core.schedules", "DecreasingProgressScheduleConfig"),
    "IncreasingProgressScheduleConfig": ("noether.core.schedules", "IncreasingProgressScheduleConfig"),
    "LinearWarmupCosineDecayScheduleConfig": (
        "noether.core.schedules",
        "LinearWarmupCosineDecayScheduleConfig",
    ),
    "PolynomialDecreasingScheduleConfig": ("noether.core.schedules", "PolynomialDecreasingScheduleConfig"),
    "PolynomialIncreasingScheduleConfig": ("noether.core.schedules", "PolynomialIncreasingScheduleConfig"),
    "ProgressScheduleConfig": ("noether.core.schedules", "ProgressScheduleConfig"),
    "ScheduleBaseConfig": ("noether.core.schedules", "ScheduleBaseConfig"),
    "SchedulerConfig": ("noether.core.schedules", "SchedulerConfig"),
    "StepDecreasingScheduleConfig": ("noether.core.schedules", "StepDecreasingScheduleConfig"),
    "StepFixedScheduleConfig": ("noether.core.schedules", "StepFixedScheduleConfig"),
    "StepIntervalScheduleConfig": ("noether.core.schedules", "StepIntervalScheduleConfig"),
    # schema
    "ConfigSchema": ("noether.core.schemas.schema", "ConfigSchema"),
    # slurm
    "SlurmConfig": ("noether.training.cli.submit_job", "SlurmConfig"),
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
    warnings.warn(
        f"Importing `{name}` from `{__name__}` is deprecated; import from `{module_path}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    from noether.core.optimizer.schemas import (
        AdamOptimizerConfig,
        AnyOptimizerConfig,
        MuonOptimizerConfig,
        OptimizerConfig,
        ParamGroupModifierConfig,
        SGDOptimizerConfig,
    )
    from noether.core.schedules import AnyScheduleConfig
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
    from noether.core.schemas.schema import ConfigSchema
    from noether.core.trackers import WandBTrackerSchema
    from noether.data.base.dataset import DatasetBaseConfig, StandardDatasetConfig
    from noether.data.preprocessors.normalizers import AnyNormalizer, FieldNormalizerConfig
    from noether.training.callbacks.offline_loss import OfflineLossCallbackConfig
    from noether.training.cli.submit_job import SlurmConfig
    from noether.training.trainers.base import BaseTrainerConfig
