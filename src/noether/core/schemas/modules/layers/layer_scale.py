#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-export for ``LayerScaleConfig``.

The config has moved next to its matching class in
:mod:`noether.modeling.modules.layers.layer_scale`.
"""

from noether.modeling.modules.layers.layer_scale import LayerScaleConfig

__all__ = ["LayerScaleConfig"]
