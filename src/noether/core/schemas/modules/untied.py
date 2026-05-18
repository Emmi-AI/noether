#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for ``noether.core.schemas.modules.untied``.

Untied configs have moved next to their matching classes in
:mod:`noether.modeling.modules.untied`.
"""

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
