#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for tracker schemas.

The canonical home is :mod:`noether.core.trackers`.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from noether.core.trackers import (
        AnyTracker,
        TensorboardTrackerSchema,
        TrackioTrackerSchema,
        WandBTrackerSchema,
    )

__all__ = [
    "AnyTracker",
    "TensorboardTrackerSchema",
    "TrackioTrackerSchema",
    "WandBTrackerSchema",
]

_LAZY: dict[str, str] = dict.fromkeys(__all__, "noether.core.trackers")


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
