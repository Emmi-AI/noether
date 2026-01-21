#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from noether.core.schemas.dataset import AeroDataSpecs
from noether.core.schemas.models.base import ModelBaseConfig
from noether.core.schemas.modules.blocks import TransformerBlockConfig


class CompositeTransformerBlockConfig(ModelBaseConfig):
    kind: Literal["tutorial.models.composite_components.CompositeTransformerBlockModel"] = (
        "tutorial.models.composite_components.CompositeTransformerBlockModel"
    )
    depth: int
    transformer_config: TransformerBlockConfig | None = None
    output_dim: int | None = None
    use_rope: bool = False
    projection_bias: bool = False
    use_output_projection: bool = False


class CompositeTransformerConfig(ModelBaseConfig):
    kind: str = "tutorial.models.composite_transformer.CompositeTransformer"
    name: Literal["composite_transformer"] = "composite_transformer"
    use_rope: bool = True
    low_level_blocks: CompositeTransformerBlockConfig
    high_level_blocks: CompositeTransformerBlockConfig
    num_heads: int | None = None
    hidden_dim: int | None = None
    mlp_expansion_factor: int | None = None
    data_specs: AeroDataSpecs
    """Data specifications for the model. If None, default data specifications will be used."""
