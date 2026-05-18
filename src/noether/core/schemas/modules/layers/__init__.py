#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for ``noether.core.schemas.modules.layers``.

Layer configs have been moved next to their matching classes in
:mod:`noether.modeling.modules.layers`. This module preserves the old
import paths.
"""

from noether.modeling.modules.layers.continuous_sincos_embed import ContinuousSincosEmbeddingConfig
from noether.modeling.modules.layers.drop_path import UnquantizedDropPathConfig
from noether.modeling.modules.layers.layer_scale import LayerScaleConfig
from noether.modeling.modules.layers.linear_projection import LinearProjectionConfig
from noether.modeling.modules.layers.rope_frequency import RopeFrequencyConfig

__all__ = [
    "ContinuousSincosEmbeddingConfig",
    "UnquantizedDropPathConfig",
    "LayerScaleConfig",
    "LinearProjectionConfig",
    "RopeFrequencyConfig",
]
