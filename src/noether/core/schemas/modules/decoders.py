#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for ``noether.core.schemas.modules.decoders``.

Decoder configs have moved next to their matching classes in
:mod:`noether.modeling.modules.decoders`.
"""

from noether.modeling.modules.decoders.deep_perceiver import DeepPerceiverDecoderConfig

__all__ = [
    "DeepPerceiverDecoderConfig",
]
