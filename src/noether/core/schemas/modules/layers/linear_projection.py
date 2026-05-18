#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-export for ``LinearProjectionConfig``.

The config has moved next to its matching class in
:mod:`noether.modeling.modules.layers.linear_projection`.
"""

from noether.modeling.modules.layers.linear_projection import LinearProjectionConfig

__all__ = ["LinearProjectionConfig"]
