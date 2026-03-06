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

    depth: int = Field(..., ge=1)

    transformer_block_config: Annotated[TransformerBlockConfig, Shared]
