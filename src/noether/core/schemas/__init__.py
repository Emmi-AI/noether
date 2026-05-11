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
    # callbacks
    "BestCheckpointCallbackConfig": ("noether.core.schemas.callbacks", "BestCheckpointCallbackConfig"),
    "BestMetricCallbackConfig": ("noether.core.schemas.callbacks", "BestMetricCallbackConfig"),
    "CallBackBaseConfig": ("noether.core.schemas.callbacks", "CallBackBaseConfig"),
    "CallbacksConfig": ("noether.core.schemas.callbacks", "CallbacksConfig"),
    "CheckpointCallbackConfig": ("noether.core.schemas.callbacks", "CheckpointCallbackConfig"),
    "EmaCallbackConfig": ("noether.core.schemas.callbacks", "EmaCallbackConfig"),
    "FixedEarlyStopperConfig": ("noether.core.schemas.callbacks", "FixedEarlyStopperConfig"),
    "MetricEarlyStopperConfig": ("noether.core.schemas.callbacks", "MetricEarlyStopperConfig"),
    "OfflineLossCallbackConfig": ("noether.core.schemas.callbacks", "OfflineLossCallbackConfig"),
    "OnlineLossCallbackConfig": ("noether.core.schemas.callbacks", "OnlineLossCallbackConfig"),
    "PeriodicDataIteratorCallbackConfig": ("noether.core.schemas.callbacks", "PeriodicDataIteratorCallbackConfig"),
    "TrackAdditionalOutputsCallbackConfig": ("noether.core.schemas.callbacks", "TrackAdditionalOutputsCallbackConfig"),
    # dataset
    "DatasetBaseConfig": ("noether.core.schemas.dataset", "DatasetBaseConfig"),
    "StandardDatasetConfig": ("noether.core.schemas.dataset", "StandardDatasetConfig"),
    # initializers
    "AnyInitializer": ("noether.core.schemas.initializers", "AnyInitializer"),
    "CheckpointInitializerConfig": ("noether.core.schemas.initializers", "CheckpointInitializerConfig"),
    "InitializerConfig": ("noether.core.schemas.initializers", "InitializerConfig"),
    "PreviousRunInitializerConfig": ("noether.core.schemas.initializers", "PreviousRunInitializerConfig"),
    "ResumeInitializerConfig": ("noether.core.schemas.initializers", "ResumeInitializerConfig"),
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
    from noether.core.models.base import ModelBaseConfig
    from noether.core.schemas.callbacks import (
        BestCheckpointCallbackConfig,
        BestMetricCallbackConfig,
        CallBackBaseConfig,
        CallbacksConfig,
        CheckpointCallbackConfig,
        EmaCallbackConfig,
        FixedEarlyStopperConfig,
        MetricEarlyStopperConfig,
        OfflineLossCallbackConfig,
        OnlineLossCallbackConfig,
        PeriodicDataIteratorCallbackConfig,
        TrackAdditionalOutputsCallbackConfig,
    )
    from noether.core.schemas.dataset import DatasetBaseConfig, StandardDatasetConfig
    from noether.core.schemas.initializers import (
        AnyInitializer,
        CheckpointInitializerConfig,
        InitializerConfig,
        PreviousRunInitializerConfig,
        ResumeInitializerConfig,
    )
    from noether.core.schemas.normalizers import AnyNormalizer, FieldNormalizerConfig
    from noether.core.schemas.optimizers import (
        AdamOptimizerConfig,
        AnyOptimizerConfig,
        MuonOptimizerConfig,
        OptimizerConfig,
        ParamGroupModifierConfig,
        SGDOptimizerConfig,
    )
    from noether.core.schemas.schedules import (
        AnyScheduleConfig,
        ConstantScheduleConfig,
        CustomScheduleConfig,
        DecreasingProgressScheduleConfig,
        IncreasingProgressScheduleConfig,
        LinearWarmupCosineDecayScheduleConfig,
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
    from noether.core.schemas.schema import ConfigSchema
    from noether.core.schemas.slurm import SlurmConfig
    from noether.core.trackers import WandBTrackerSchema
    from noether.training.trainers.base import BaseTrainerConfig
