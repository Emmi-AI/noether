#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for ``noether.core.schemas.modules.blocks``.

Block configs have moved next to their matching classes in
:mod:`noether.modeling.modules.blocks`.
"""

from noether.modeling.modules.blocks.perceiver import PerceiverBlockConfig
from noether.modeling.modules.blocks.transformer import TransformerBlockConfig

__all__ = [
    "PerceiverBlockConfig",
    "TransformerBlockConfig",
]
