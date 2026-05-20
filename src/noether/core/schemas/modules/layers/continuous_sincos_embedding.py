#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-export for ``ContinuousSincosEmbeddingConfig``.

The canonical home is :mod:`noether.modeling.modules.layers.continuous_sincos_embed`.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from noether.modeling.modules.layers.continuous_sincos_embed import ContinuousSincosEmbeddingConfig

__all__ = ["ContinuousSincosEmbeddingConfig"]

_LAZY: dict[str, str] = {
    "ContinuousSincosEmbeddingConfig": "noether.modeling.modules.layers.continuous_sincos_embed",
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
