#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-export for ``BaseTrainerConfig``.

The canonical home is :mod:`noether.training.trainers.base`.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from noether.training.trainers.base import BaseTrainerConfig

__all__ = ["BaseTrainerConfig"]

_LAZY: dict[str, str] = {"BaseTrainerConfig": "noether.training.trainers.base"}


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
