#  Copyright © 2026 Emmi AI GmbH. All rights reserved.
"""Back-compat re-export for ``VectorsConditionerConfig``.

The config has moved next to its matching class in
:mod:`noether.modeling.modules.layers.vectors_conditioner`.
"""

from noether.modeling.modules.layers.vectors_conditioner import VectorsConditionerConfig

__all__ = ["VectorsConditionerConfig"]
