#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Annotated

from pydantic import BaseModel, Field

from noether.core.schemas.mixins import InjectSharedFieldFromParentMixin, Shared
from noether.core.schemas.modules.blocks import PerceiverBlockConfig


class DeepPerceiverDecoderConfig(InjectSharedFieldFromParentMixin, BaseModel):
    """Configuration for the DeepPerceiverDecoder module."""

    perceiver_block_config: Annotated[PerceiverBlockConfig, Shared] = Field(...)
    """Configuration for the Perceiver blocks used in the decoder."""

    depth: int = Field(1, ge=1)
    """Number of deep perceiver decoder layers (i.e., depth of the network). Defaults to 1."""

    input_dim: int = Field(..., ge=1)
    """Input dimension for the query positions."""
