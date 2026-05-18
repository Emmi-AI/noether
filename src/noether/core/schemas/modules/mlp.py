#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for ``noether.core.schemas.modules.mlp``.

MLP configs have moved next to their matching classes in
:mod:`noether.modeling.modules.mlp`.
"""

from noether.modeling.modules.mlp.mlp import MLPConfig
from noether.modeling.modules.mlp.upactdown_mlp import UpActDownMLPConfig

__all__ = [
    "MLPConfig",
    "UpActDownMLPConfig",
]
