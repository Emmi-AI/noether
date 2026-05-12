#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Base normalizer schema, helpers, and re-exports.

The base ``NormalizerConfig`` and the tensor helpers stay here because
``noether.core.schemas.dataset`` consumes them at module load time, and
relocating them under ``noether.data.*`` would force ``noether/data/__init__``
to run before ``core/schemas/dataset`` can finish loading (circular import).

The concrete ``*NormalizerConfig`` classes have been moved next to their
matching classes in :mod:`noether.data.preprocessors.normalizers`. They are
re-exported lazily here for backward compatibility.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, PlainSerializer, PlainValidator

from noether.core.schemas.lib import _RegistryBase


def validate_tensor(v: Any) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    try:
        return torch.tensor(v)
    except Exception as e:
        raise ValueError(f"Could not convert {v} to torch.Tensor: {e}") from None


TorchTensor = Annotated[
    torch.Tensor,
    PlainValidator(validate_tensor),
    PlainSerializer(lambda x: x.tolist(), return_type=list, when_used="always"),
]

FloatOrArray = float | Sequence[float] | TorchTensor
SequenceOrTensor = Sequence[float] | TorchTensor


class NormalizerConfig(_RegistryBase):
    """Base configuration for normalizers. All normalizer configs should inherit from this class."""

    _registry: ClassVar[dict[str, type[BaseModel]]] = {}
    _type_field: ClassVar[str] = "kind"
    kind: str | None = None
    """Kind of normalizer to use, i.e. class path"""

    model_config = ConfigDict(extra="forbid")


if TYPE_CHECKING:
    from noether.data.preprocessors.normalizers import (
        FieldNormalizerConfig,
        MeanStdNormalizerConfig,
        PositionNormalizerConfig,
        ShiftAndScaleNormalizerConfig,
    )

    AnyNormalizer = Union[
        MeanStdNormalizerConfig, PositionNormalizerConfig, ShiftAndScaleNormalizerConfig, FieldNormalizerConfig
    ]

_LAZY: dict[str, str] = {
    "FieldNormalizerConfig": "FieldNormalizerConfig",
    "MeanStdNormalizerConfig": "MeanStdNormalizerConfig",
    "PositionNormalizerConfig": "PositionNormalizerConfig",
    "ShiftAndScaleNormalizerConfig": "ShiftAndScaleNormalizerConfig",
}

__all__ = [
    "AnyNormalizer",
    "FieldNormalizerConfig",
    "FloatOrArray",
    "MeanStdNormalizerConfig",
    "NormalizerConfig",
    "PositionNormalizerConfig",
    "SequenceOrTensor",
    "ShiftAndScaleNormalizerConfig",
    "TorchTensor",
    "validate_tensor",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = importlib.import_module("noether.data.preprocessors.normalizers")
        value = getattr(module, _LAZY[name])
        globals()[name] = value
        return value
    if name == "AnyNormalizer":
        module = importlib.import_module("noether.data.preprocessors.normalizers")
        value = Union[
            module.MeanStdNormalizerConfig,
            module.PositionNormalizerConfig,
            module.ShiftAndScaleNormalizerConfig,
            module.FieldNormalizerConfig,
        ]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
