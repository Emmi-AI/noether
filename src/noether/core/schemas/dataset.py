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

from __future__ import annotations

import importlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, RootModel, model_validator


class FieldDimSpec(RootModel[OrderedDict[str, int]]):
    """A specification for a group of named data fields and their dimensions."""

    @property
    def field_slices(self) -> dict[str, slice]:
        """Calculates slice indices for each field in concatenation order."""
        indices = {}
        start = 0
        for field, dim in self.root.items():
            if not isinstance(dim, int) or dim <= 0:
                continue
            indices[field] = slice(start, start + dim)
            start += dim
        return indices

    @property
    def total_dim(self) -> int:
        """Calculates the total dimension of all fields combined."""
        return sum(self.root.values())

    def __getitem__(self, key: str) -> int:
        return self.root[key]

    def __iter__(self):
        return iter(self.root.items())

    def __getattr__(self, name: str) -> int:
        """Enables attribute-style access (e.g., `spec.geometry`)."""
        try:
            return self.root[name]
        except KeyError as err:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from err

    def __dir__(self) -> list[str]:
        """Improves autocompletion for dynamic attributes."""
        return sorted(set(super().__dir__()) | set(self.root.keys()))

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()

    def items(self):
        return self.root.items()

    def __len__(self):
        return len(self.root)


class DomainDataSpec(BaseModel):
    """Data specification for a single domain (e.g., surface, volume, wake)."""

    output_dims: FieldDimSpec
    """Output fields and their dimensions for this domain, e.g. {"pressure": 1, "velocity": 3}."""
    feature_dim: FieldDimSpec | None = None
    """Input feature fields and their dimensions for this domain."""


class ModelDataSpecs(BaseModel):
    """Base data specification for models that operate on arbitrary named domains.

    This is the minimal interface that model configs need from data specifications:
    position dimensions, available conditioning, and per-domain data descriptions.
    """

    position_dim: int = Field(..., ge=1)
    """Dimension of the input position vectors."""
    conditioning_dims: FieldDimSpec | None = None
    """Available conditioning features and their dimensions."""
    domains: dict[str, DomainDataSpec] = Field(default_factory=dict)
    """Per-domain data specifications keyed by domain name."""
    use_physics_features: bool = True
    """Whether physics features are used as input."""

    @property
    def total_output_dim(self) -> int:
        """Calculates the total output dimension across all domains."""
        return sum(spec.output_dims.total_dim for spec in self.domains.values())

    @property
    def all_targets(self) -> set[str]:
        """Returns all target field names across all domains, prefixed by domain name."""
        targets: set[str] = set()
        for name, spec in self.domains.items():
            targets |= {f"{name}_{key}" for key in spec.output_dims.keys()}
        return targets

    @property
    def all_features(self) -> set[str]:
        """Returns all feature field names across all domains."""
        features: set[str] = set()
        for spec in self.domains.values():
            if spec.feature_dim:
                features |= set(spec.feature_dim.keys())
        return features

    @model_validator(mode="after")
    def remove_feature_fields(self):
        if not self.use_physics_features:
            for spec in self.domains.values():
                spec.feature_dim = None
        return self


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
    value = getattr(importlib.import_module(module_path), attr)
    globals()[name] = value
    return value


if TYPE_CHECKING:  # static type checkers — keep in sync with _LAZY
    from noether.data.base.dataset import DatasetBaseConfig, DatasetSplitIDs, StandardDatasetConfig
    from noether.data.base.wrapper import DatasetWrapperConfig
    from noether.data.base.wrappers import DatasetWrappers
    from noether.data.base.wrappers.repeat import RepeatWrapperConfig
    from noether.data.base.wrappers.shuffle import ShuffleWrapperConfig
    from noether.data.base.wrappers.subset import SubsetWrapperConfig
    from noether.data.pipeline.multistage import PipelineConfig
