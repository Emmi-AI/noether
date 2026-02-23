#  Copyright © 2025 Emmi AI GmbH. All rights reserved.


from pydantic import ConfigDict, Field

from noether.core.schemas.dataset import AeroDataSpecs
from noether.core.schemas.mixins import SharedFieldPropagationMixin
from noether.core.schemas.modules import DeepPerceiverDecoderConfig, SupernodePoolingConfig
from noether.core.schemas.modules.blocks import TransformerBlockConfig

from .base import ModelBaseConfig


class UPTConfig(ModelBaseConfig, SharedFieldPropagationMixin):
    """Configuration for a Transolver model."""

    model_config = ConfigDict(extra="forbid")

    _SHARED_CONFIGS_MAP = {
        "supernode_pooling_config": SupernodePoolingConfig,
        "approximator_config": TransformerBlockConfig,
        "decoder_config": DeepPerceiverDecoderConfig,
    }

    num_heads: int = Field(..., ge=1)
    """Number of attention heads in the model."""

    hidden_dim: int = Field(..., ge=1)
    """Hidden dimension of the model."""

    mlp_expansion_factor: int = Field(..., ge=1)
    """Expansion factor for the MLP of the FF layers."""

    approximator_depth: int = Field(..., ge=1)
    """Number of approximator layers."""

    use_rope: bool = Field(False)

    supernode_pooling_config: SupernodePoolingConfig

    approximator_config: TransformerBlockConfig

    decoder_config: DeepPerceiverDecoderConfig

    bias_layers: bool = Field(False)

    data_specs: AeroDataSpecs
