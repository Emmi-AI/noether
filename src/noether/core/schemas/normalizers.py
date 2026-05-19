#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for normalizer configs.

The canonical home is :mod:`noether.data.preprocessors.normalizers`.
"""

from __future__ import annotations

import importlib
import warnings
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

_CANONICAL = "noether.data.preprocessors.normalizers"
_LAZY: dict[str, str] = dict.fromkeys(
    [n for n in __all__ if n != "AnyNormalizer"],
    _CANONICAL,
)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        warnings.warn(
            f"Importing `{name}` from `{__name__}` is deprecated; import from `{_CANONICAL}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(_CANONICAL), name)
    if name == "AnyNormalizer":
        warnings.warn(
            f"Importing `AnyNormalizer` from `{__name__}` is deprecated; import from `{_CANONICAL}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = importlib.import_module(_CANONICAL)
        return Union[
            module.MeanStdNormalizerConfig,
            module.PositionNormalizerConfig,
            module.ShiftAndScaleNormalizerConfig,
            module.FieldNormalizerConfig,
        ]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
