#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-export for ``RopeFrequencyConfig``.

The config has moved next to its matching class in
:mod:`noether.modeling.modules.layers.rope_frequency`.
"""

from noether.modeling.modules.layers.rope_frequency import RopeFrequencyConfig

__all__ = ["RopeFrequencyConfig"]
