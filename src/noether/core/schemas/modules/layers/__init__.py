#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for layer configs.

Canonical homes are under :mod:`noether.modeling.modules.layers`.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from noether.modeling.modules.layers.continuous_sincos_embed import ContinuousSincosEmbeddingConfig
    from noether.modeling.modules.layers.drop_path import UnquantizedDropPathConfig
    from noether.modeling.modules.layers.layer_scale import LayerScaleConfig
    from noether.modeling.modules.layers.linear_projection import LinearProjectionConfig
    from noether.modeling.modules.layers.rope_frequency import RopeFrequencyConfig

__all__ = [
    "ContinuousSincosEmbeddingConfig",
    "UnquantizedDropPathConfig",
    "LayerScaleConfig",
    "LinearProjectionConfig",
    "RopeFrequencyConfig",
]

_LAZY: dict[str, str] = {
    "ContinuousSincosEmbeddingConfig": "noether.modeling.modules.layers.continuous_sincos_embed",
    "UnquantizedDropPathConfig": "noether.modeling.modules.layers.drop_path",
    "LayerScaleConfig": "noether.modeling.modules.layers.layer_scale",
    "LinearProjectionConfig": "noether.modeling.modules.layers.linear_projection",
    "RopeFrequencyConfig": "noether.modeling.modules.layers.rope_frequency",
}


def __getattr__(name: str) -> Any:
    try:
        module_path = _LAZY[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    warnings.warn(
        f"Importing `{name}` from `{__name__}` is deprecated; import from `{module_path}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(module_path), name)
