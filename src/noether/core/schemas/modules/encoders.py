#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for ``noether.core.schemas.modules.encoders``.

Encoder configs have moved next to their matching classes in
:mod:`noether.modeling.modules.encoders`.
"""

from noether.modeling.modules.encoders.supernode_pooling import SupernodePoolingConfig

__all__ = [
    "SupernodePoolingConfig",
]
