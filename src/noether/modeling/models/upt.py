#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Annotated

import torch
from pydantic import ConfigDict, Field, computed_field, model_validator
from torch import nn

from noether.core.models.base import ModelBaseConfig
from noether.core.schemas.mixins import InjectSharedFieldFromParentMixin, Shared
from noether.data.schemas import ModelDataSpecs
from noether.modeling.modules import DeepPerceiverDecoder, SupernodePooling, TransformerBlock
from noether.modeling.modules.blocks.transformer import TransformerBlockConfig
from noether.modeling.modules.decoders.deep_perceiver import DeepPerceiverDecoderConfig
from noether.modeling.modules.encoders.supernode_pooling import SupernodePoolingConfig
from noether.modeling.modules.layers import ContinuousSincosEmbed, LinearProjection, RopeFrequency
from noether.modeling.modules.layers.continuous_sincos_embed import ContinuousSincosEmbeddingConfig
from noether.modeling.modules.layers.linear_projection import LinearProjectionConfig
from noether.modeling.modules.layers.rope_frequency import RopeFrequencyConfig


class UPTConfig(ModelBaseConfig, InjectSharedFieldFromParentMixin):
    """Configuration for a UPT model."""

    model_config = ConfigDict(extra="forbid")

    num_heads: int = Field(..., ge=1)
    """Number of attention heads in the model."""

    hidden_dim: int = Field(..., ge=1)
    """Hidden dimension of the model."""

    mlp_expansion_factor: int = Field(..., ge=1)
    """Expansion factor for the MLP of the FF layers."""

    approximator_depth: int = Field(..., ge=1)
    """Number of approximator layers."""

    use_rope: bool = Field(False)

    bias: bool = Field(True)
    """Whether to use bias terms in the model's linear layers."""

    supernode_pooling_config: Annotated[SupernodePoolingConfig, Shared]

    approximator_config: Annotated[TransformerBlockConfig, Shared]

    decoder_config: Annotated[DeepPerceiverDecoderConfig, Shared]

    bias_layers: bool = Field(False)

    data_specs: ModelDataSpecs

    @computed_field
    def linear_output_projection_config(self) -> "LinearProjectionConfig":
        return LinearProjectionConfig(
            input_dim=self.hidden_dim,
            output_dim=self.data_specs.total_output_dim,
            init_weights=self.decoder_config.perceiver_block_config.init_weights,
            bias=self.bias,
        )

    @computed_field
    def rope_frequency_config(self) -> "RopeFrequencyConfig":
        return RopeFrequencyConfig(
            hidden_dim=self.hidden_dim // self.num_heads,
            input_dim=self.data_specs.position_dim,
            implementation="complex",
            max_wavelength=self.approximator_config.max_wavelength,
        )

    @model_validator(mode="after")
    def validate_rope_usage(self) -> "UPTConfig":
        """Ensure that if use_rope is True in the main config, it is also True in the approximator_config."""
        if self.use_rope:
            if not (self.approximator_config.use_rope and self.decoder_config.perceiver_block_config.use_rope):
                raise ValueError(
                    "If 'use_rope' is set to True in the UPTConfig, it must also be set to True in the approximator_config."
                )
        return self

    @computed_field
    def pos_embedding_config(self) -> ContinuousSincosEmbeddingConfig:
        return ContinuousSincosEmbeddingConfig(
            hidden_dim=self.hidden_dim,
            input_dim=self.data_specs.position_dim,
            max_wavelength=self.approximator_config.max_wavelength,
        )

    @model_validator(mode="after")
    def validate_parameters(self) -> "UPTConfig":
        """Validate validity of parameters across the model and its submodules.

        Ensures that:
        1. hidden_dim is divisible by num_heads in parent and all submodules with num_heads
        2. hidden_dim is consistent across parent and all submodules
        """
        # 1. Parent check: hidden_dim % num_heads == 0
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads}).")

        # 2. SupernodePoolingConfig: hidden_dim equality
        if self.supernode_pooling_config.hidden_dim != self.hidden_dim:
            raise ValueError(
                f"supernode_pooling_config.hidden_dim ({self.supernode_pooling_config.hidden_dim}) "
                f"must match model hidden_dim ({self.hidden_dim})."
            )

        # 3. ApproximatorConfig: hidden_dim equality + modulo check
        if self.approximator_config.hidden_dim != self.hidden_dim:
            raise ValueError(
                f"approximator_config.hidden_dim ({self.approximator_config.hidden_dim}) "
                f"must match model hidden_dim ({self.hidden_dim})."
            )

        if self.approximator_config.hidden_dim % self.approximator_config.num_heads != 0:
            raise ValueError(
                f"approximator_config.hidden_dim ({self.approximator_config.hidden_dim}) "
                f"must be divisible by approximator_config.num_heads ({self.approximator_config.num_heads})."
            )

        # 4. DecoderConfig: check nested perceiver_block_config
        perceiver_config = self.decoder_config.perceiver_block_config

        if perceiver_config.hidden_dim != self.hidden_dim:
            raise ValueError(
                f"decoder_config.perceiver_block_config.hidden_dim ({perceiver_config.hidden_dim}) "
                f"must match model hidden_dim ({self.hidden_dim})."
            )

        if perceiver_config.hidden_dim % perceiver_config.num_heads != 0:
            raise ValueError(
                f"decoder_config.perceiver_block_config.hidden_dim ({perceiver_config.hidden_dim}) "
                f"must be divisible by decoder_config.perceiver_block_config.num_heads ({perceiver_config.num_heads})."
            )

        return self


class UPT(nn.Module):
    """Implementation of the UPT (Universal Physics Transformer) model."""

    def __init__(
        self,
        config: UPTConfig,
    ):
        """
        Args:
            config: Configuration for the UPT model. See :class:`~noether.core.schemas.models.UPTConfig` for details.
        """

        super().__init__()

        self.use_rope = config.use_rope
        self.encoder = SupernodePooling(config=config.supernode_pooling_config)
        self.pos_embed = ContinuousSincosEmbed(config=config.pos_embedding_config)  # type: ignore[arg-type]
        if self.use_rope:
            self.rope = RopeFrequency(config=config.rope_frequency_config)  # type: ignore[arg-type]

        self.approximator_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config=config.approximator_config,
                )
                for _ in range(config.approximator_depth)
            ],
        )

        self.decoder = DeepPerceiverDecoder(config=config.decoder_config)  # type: ignore[arg-type]

        self.norm = nn.RMSNorm(
            config.decoder_config.perceiver_block_config.hidden_dim,
            eps=config.decoder_config.perceiver_block_config.eps,
        )

        self.prediction_layer = LinearProjection(config=config.linear_output_projection_config)  # type: ignore[arg-type]

    def compute_rope_args(
        self,
        geometry_batch_idx: torch.Tensor,
        geometry_position: torch.Tensor,
        geometry_supernode_idx: torch.Tensor,
        query_position: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute the RoPE frequency arguments for the geometry and query positions.
        If RoPE is not used, return empty dicts.
        """
        if not self.use_rope:
            return {}, {}

        batch_size = geometry_batch_idx.unique().shape[0]
        supernode_freqs = self.rope(geometry_position[geometry_supernode_idx])
        channels = supernode_freqs.shape[-1]
        if supernode_freqs.ndim == 2:
            supernode_freqs = supernode_freqs.unsqueeze(0)  # add batch dimension
        supernode_freqs = supernode_freqs.reshape(batch_size, -1, channels)
        encoder_attn_kwargs = dict(freqs=supernode_freqs)
        decoder_attn_kwargs = dict(
            q_freqs=self.rope(query_position),
            k_freqs=supernode_freqs,
        )

        return encoder_attn_kwargs, decoder_attn_kwargs

    def forward(
        self,
        geometry_batch_idx: torch.Tensor,
        geometry_supernode_idx: torch.Tensor,
        geometry_position: torch.Tensor,
        query_position: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the UPT model.

        Args:
            geometry_batch_idx: Batch indices for the geometry positions.
            geometry_supernode_idx: Supernode indices for the geometry positions.
            geometry_position: Input coordinates of the geometry mesh points.
            query_position: Input coordinates of the query points.

        Returns:
            torch.Tensor: Output tensor containing the predictions at query positions.
        """

        encoder_attn_kwargs, decoder_attn_kwargs = self.compute_rope_args(
            geometry_batch_idx, geometry_position, geometry_supernode_idx, query_position
        )

        # supernode pooling encoder
        x = self.encoder(
            input_pos=geometry_position,
            supernode_idx=geometry_supernode_idx,
            batch_idx=geometry_batch_idx,
        )
        # approximator blocks
        for block in self.approximator_blocks:
            x, _ = block(x, attn_kwargs=encoder_attn_kwargs)

        queries = self.pos_embed(query_position)

        # perceiver decoder
        x = self.decoder(
            kv=x,
            queries=queries,
            attn_kwargs=decoder_attn_kwargs,
            condition=None,
        )

        x = self.norm(x)
        return self.prediction_layer(x)  # type: ignore[no-any-return]
