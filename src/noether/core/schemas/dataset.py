#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Dataset-related shared types and back-compat re-exports.

The dataset, wrapper, and pipeline config classes have moved next to the
classes they configure under :mod:`noether.data`. They are re-exported here
lazily to keep the old ``from noether.core.schemas.dataset import X`` paths
working without eagerly triggering the heavy ``noether.data`` imports.

The model-data specification types (:class:`FieldDimSpec`,
:class:`DomainDataSpec`, :class:`ModelDataSpecs`) remain here because they
have no single implementation home and are consumed across model configs.
"""

import importlib
import warnings
from typing import TYPE_CHECKING, Any

# Back-compat: lazy re-exports for config classes moved to noether.data.*
_LAZY: dict[str, tuple[str, str]] = {
    "DatasetWrapperConfig": ("noether.data.base.wrapper", "DatasetWrapperConfig"),
    "RepeatWrapperConfig": ("noether.data.base.wrappers.repeat", "RepeatWrapperConfig"),
    "ShuffleWrapperConfig": ("noether.data.base.wrappers.shuffle", "ShuffleWrapperConfig"),
    "SubsetWrapperConfig": ("noether.data.base.wrappers.subset", "SubsetWrapperConfig"),
    "DatasetWrappers": ("noether.data.base.wrappers", "DatasetWrappers"),
    "PipelineConfig": ("noether.data.pipeline.multistage", "PipelineConfig"),
    "DatasetBaseConfig": ("noether.data.base.dataset", "DatasetBaseConfig"),
    "StandardDatasetConfig": ("noether.data.base.dataset", "StandardDatasetConfig"),
    "DatasetSplitIDs": ("noether.data.base.dataset", "DatasetSplitIDs"),
    "FieldDimSpec": ("noether.data.schemas", "FieldDimSpec"),
    "DomainDataSpec": ("noether.data.schemas", "DomainDataSpec"),
    "ModelDataSpecs": ("noether.data.schemas", "ModelDataSpecs"),
}

__all__ = [
    "DatasetBaseConfig",
    "DatasetSplitIDs",
    "DatasetWrapperConfig",
    "DatasetWrappers",
    "DomainDataSpec",
    "FieldDimSpec",
    "ModelDataSpecs",
    "PipelineConfig",
    "RepeatWrapperConfig",
    "ShuffleWrapperConfig",
    "StandardDatasetConfig",
    "SubsetWrapperConfig",
]


def __getattr__(name: str) -> Any:
    try:
        module_path, attr = _LAZY[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    warnings.warn(
        f"Importing `{name}` from `{__name__}` is deprecated; import from `{module_path}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(module_path), attr)


if TYPE_CHECKING:  # static type checkers — keep in sync with _LAZY
    from noether.data.base.dataset import DatasetBaseConfig, DatasetSplitIDs, StandardDatasetConfig
    from noether.data.base.wrapper import DatasetWrapperConfig
    from noether.data.base.wrappers import DatasetWrappers
    from noether.data.base.wrappers.repeat import RepeatWrapperConfig
    from noether.data.base.wrappers.shuffle import ShuffleWrapperConfig
    from noether.data.base.wrappers.subset import SubsetWrapperConfig
    from noether.data.pipeline.multistage import PipelineConfig
    from noether.data.schemas import DomainDataSpec, FieldDimSpec, ModelDataSpecs
