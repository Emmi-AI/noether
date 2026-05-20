#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for untied configs.

The canonical home is :mod:`noether.modeling.modules.untied`.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from noether.modeling.modules.untied import (
        UntiedLinearConfig,
        UntiedMixedAttentionConfig,
        UntiedMLPConfig,
        UntiedPerceiverBlockConfig,
        UntiedTransformerBlockConfig,
    )

__all__ = [
    "UntiedLinearConfig",
    "UntiedMLPConfig",
    "UntiedMixedAttentionConfig",
    "UntiedPerceiverBlockConfig",
    "UntiedTransformerBlockConfig",
]

_LAZY: dict[str, str] = dict.fromkeys(__all__, "noether.modeling.modules.untied")


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
