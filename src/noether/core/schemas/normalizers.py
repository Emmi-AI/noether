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
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from noether.data.preprocessors.normalizers import (
        AnyNormalizer,
        FieldNormalizerConfig,
        FloatOrArray,
        MeanStdNormalizerConfig,
        NormalizerConfig,
        PositionNormalizerConfig,
        SequenceOrTensor,
        ShiftAndScaleNormalizerConfig,
        TorchTensor,
        validate_tensor,
    )


_LAZY: dict[str, str] = {
    "FieldNormalizerConfig": "FieldNormalizerConfig",
    "MeanStdNormalizerConfig": "MeanStdNormalizerConfig",
    "PositionNormalizerConfig": "PositionNormalizerConfig",
    "ShiftAndScaleNormalizerConfig": "ShiftAndScaleNormalizerConfig",
    "NormalizerConfig": "NormalizerConfig",
    "TorchTensor": "TorchTensor",
    "SequenceOrTensor": "SequenceOrTensor",
    "FloatOrArray": "FloatOrArray",
    "validate_tensor": "validate_tensor",
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
