#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for ``noether.core.schemas.models``.

Model configs now live alongside their model classes under
``noether.modeling.models.*`` (and ``noether.core.models.base`` for the
base config). They are re-exported lazily via :pep:`562` so that
``import noether.core.schemas.models`` does not pull the new-home
modules into the import graph eagerly.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ModelBaseConfig": ("noether.core.models.base", "ModelBaseConfig"),
    "AnchorBranchedUPTConfig": ("noether.modeling.models.ab_upt", "AnchorBranchedUPTConfig"),
    "TransformerConfig": ("noether.modeling.models.transformer", "TransformerConfig"),
    "TransolverConfig": ("noether.modeling.models.transolver", "TransolverConfig"),
    "TransolverPlusPlusConfig": ("noether.modeling.models.transolver", "TransolverPlusPlusConfig"),
    "UPTConfig": ("noether.modeling.models.upt", "UPTConfig"),
    "ViTConfig": ("noether.modeling.models.vit", "ViTConfig"),
}


__all__ = [
    "AnchorBranchedUPTConfig",
    "ModelBaseConfig",
    "TransformerConfig",
    "TransolverConfig",
    "TransolverPlusPlusConfig",
    "UPTConfig",
    "ViTConfig",
]


def __getattr__(name: str) -> Any:
    try:
        module_path, attr = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    warnings.warn(
        f"Importing `{name}` from `{__name__}` is deprecated; import from `{module_path}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(module_path), attr)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))


if TYPE_CHECKING:  # static type checkers — keep in sync with _LAZY_EXPORTS
    from noether.core.models.base import ModelBaseConfig
    from noether.modeling.models.ab_upt import AnchorBranchedUPTConfig
    from noether.modeling.models.transformer import TransformerConfig
    from noether.modeling.models.transolver import TransolverConfig, TransolverPlusPlusConfig
    from noether.modeling.models.upt import UPTConfig
    from noether.modeling.models.vit import ViTConfig
