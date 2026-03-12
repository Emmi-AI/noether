#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Annotated

from pydantic import ConfigDict, Field

from noether.core.schemas.mixins import InjectSharedFieldFromParentMixin, Shared
from noether.core.schemas.modules.blocks import TransformerBlockConfig

from .base import ModelBaseConfig


class TransformerConfig(ModelBaseConfig, InjectSharedFieldFromParentMixin):
    """Configuration for a Transformer model."""

    model_config = ConfigDict(extra="forbid")

    hidden_dim: int = Field(..., ge=1)
    """Hidden dimension of the model. Used for all transformer blocks."""

    depth: int = Field(..., ge=1)
    """Number of transformer blocks in the model."""

    transformer_block_config: Annotated[TransformerBlockConfig, Shared]
