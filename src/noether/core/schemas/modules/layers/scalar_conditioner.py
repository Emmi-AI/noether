#  Copyright © 2026 Emmi AI GmbH. All rights reserved.
"""Back-compat re-export for ``ScalarsConditionerConfig``.

The config has moved next to its matching class in
:mod:`noether.modeling.modules.layers.scalar_conditioner`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from noether.modeling.modules.layers.scalar_conditioner import ScalarsConditionerConfig

__all__ = ["ScalarsConditionerConfig"]


def __getattr__(name: str) -> Any:
    if name == "ScalarsConditionerConfig":
        warnings.warn(
            "Importing from `noether.core.schemas.modules.layers.scalar_conditioner` is deprecated; "
            "import from `noether.modeling.modules.layers.scalar_conditioner` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from noether.modeling.modules.layers.scalar_conditioner import ScalarsConditionerConfig

        return ScalarsConditionerConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
