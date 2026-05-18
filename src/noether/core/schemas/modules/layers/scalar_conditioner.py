#  Copyright © 2026 Emmi AI GmbH. All rights reserved.
"""Back-compat re-export for ``ScalarsConditionerConfig``.

The config has moved next to its matching class in
:mod:`noether.modeling.modules.layers.scalar_conditioner`.
"""

from noether.modeling.modules.layers.scalar_conditioner import ScalarsConditionerConfig

__all__ = ["ScalarsConditionerConfig"]
