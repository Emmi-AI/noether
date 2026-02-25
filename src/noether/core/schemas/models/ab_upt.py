#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Annotated, Literal

from pydantic import ConfigDict, Field

from noether.core.schemas.dataset import AeroDataSpecs
from noether.core.schemas.mixins import InjectSharedFieldFromParentMixin, Shared
from noether.core.schemas.modules.blocks import TransformerBlockConfig
from noether.core.schemas.modules.encoders import SupernodePoolingConfig
from noether.core.types import InitWeightsMode

from .base import ModelBaseConfig


class AnchorBranchedUPTConfig(ModelBaseConfig, InjectSharedFieldFromParentMixin):
    model_config = ConfigDict(extra="forbid")

    supernode_pooling_config: Annotated[SupernodePoolingConfig, Shared]

    transformer_block_config: Annotated[TransformerBlockConfig, Shared]

    geometry_depth: int = Field(..., ge=0)
    """Number of transformer blocks in the geometry encoder."""

    hidden_dim: int = Field(..., ge=1)
    """Hidden dimension of the model."""

    physics_blocks: list[Literal["shared", "cross", "joint", "perceiver"]]
    """Types of physics blocks to use in the model.
    Options are "shared", "cross", "joint", and "perceiver".
    Shared: Self-attention within a branch (surface or volume). Attention blocks share weights between surface and volume.
    Cross: Cross-attention between surface and volume branches. Weights are shared between surface and volume.
    Joint: Joint attention over surface and volume points. I.e. full self-attention over both surface and volume points.
    Perceiver: Perceiver-style attention blocks."""

    num_surface_blocks: int = Field(..., ge=1)
    """Number of transformer blocks in the surface decoder. Weights are not shared with the volume decoder."""

    num_volume_blocks: int = Field(..., ge=1)
    """Number of transformer blocks in the volume decoder. Weights are not shared with the surface decoder."""

    init_weights: InitWeightsMode = Field("truncnormal002")
    """Weight initialization of linear layers. Defaults to "truncnormal002"."""

    drop_path_rate: float = Field(0.0)
    """Drop path rate for stochastic depth. Defaults to 0.0 (no drop path)."""

    data_specs: AeroDataSpecs
    """Data specifications for the model."""
