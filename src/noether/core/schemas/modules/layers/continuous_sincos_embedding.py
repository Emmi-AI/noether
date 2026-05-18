#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-export for ``ContinuousSincosEmbeddingConfig``.

The config has moved next to its matching class in
:mod:`noether.modeling.modules.layers.continuous_sincos_embed`.
"""

from noether.modeling.modules.layers.continuous_sincos_embed import ContinuousSincosEmbeddingConfig

__all__ = ["ContinuousSincosEmbeddingConfig"]
