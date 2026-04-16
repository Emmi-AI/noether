#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import math
from typing import Annotated

from pydantic import ConfigDict, Field, model_validator

from noether.core.schemas.mixins import InjectSharedFieldFromParentMixin, Shared
from noether.core.schemas.modules.blocks import TransformerBlockConfig
from noether.core.schemas.parameterizations import CompletePConfig

from .base import ModelBaseConfig


class TransformerConfig(ModelBaseConfig, InjectSharedFieldFromParentMixin):
    """Configuration for a Transformer model."""

    model_config = ConfigDict(extra="forbid")

    hidden_dim: int = Field(..., ge=1)
    """Hidden dimension of the model. Used for all transformer blocks."""

    depth: int = Field(..., ge=1)
    """Number of transformer blocks in the model."""

    output_scale: float = Field(1.0, gt=0.0)
    """Scaling factor applied to the output hidden states. CompleteP sets this to output_alpha / m_w. Defaults to 1.0."""

    completep_config: CompletePConfig | None = Field(None)
    """Optional CompleteP parameterization config. When set, automatically computes residual_scale, attn_scale, init_std, and output_scale."""

    transformer_block_config: Annotated[TransformerBlockConfig, Shared]

    @model_validator(mode="after")
    def apply_completep(self) -> "TransformerConfig":
        """When CompleteP config is provided, compute and inject all derived scaling factors."""
        if self.completep_config is not None:
            cfg = self.completep_config
            m_w = self.hidden_dim / cfg.base_width
            m_d = self.depth / cfg.base_depth
            head_dim = self.hidden_dim // self.transformer_block_config.num_heads

            self.transformer_block_config.residual_scale = 1.0 / (m_d**cfg.depth_alpha_exp)
            self.transformer_block_config.attn_scale = 1.0 / head_dim
            self.transformer_block_config.init_std = cfg.init_std / math.sqrt(m_w)
            self.output_scale = cfg.output_alpha / m_w
        return self
