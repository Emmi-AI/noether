#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for block configs.

Canonical homes are under :mod:`noether.modeling.modules.blocks`.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from noether.modeling.modules.blocks.perceiver import PerceiverBlockConfig
    from noether.modeling.modules.blocks.transformer import TransformerBlockConfig

__all__ = ["PerceiverBlockConfig", "TransformerBlockConfig"]

_LAZY: dict[str, str] = {
    "PerceiverBlockConfig": "noether.modeling.modules.blocks.perceiver",
    "TransformerBlockConfig": "noether.modeling.modules.blocks.transformer",
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
