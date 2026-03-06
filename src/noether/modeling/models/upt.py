#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
from torch import nn

from noether.core.schemas.models import UPTConfig
from noether.modeling.modules import DeepPerceiverDecoder, SupernodePooling, TransformerBlock
from noether.modeling.modules.layers import ContinuousSincosEmbed, LinearProjection, RopeFrequency


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

        self.encoder = SupernodePooling(config=config.supernode_pooling_config)
        self.use_rope = config.use_rope

        assert config.pos_embedding_config is not None
        self.pos_embed = ContinuousSincosEmbed(config=config.pos_embedding_config)

        if self.use_rope:
            if not config.approximator_config.use_rope and config.decoder_config.perceiver_block_config.use_rope:
                raise ValueError(
                    "If 'use_rope' is set to True in the UPTConfig, it must also be set to True in the approximator_config."
                )
            assert config.rope_frequency_config is not None
            self.rope = RopeFrequency(config=config.rope_frequency_config)

        self.approximator_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config=config.approximator_config,
                )
                for _ in range(config.approximator_depth)
            ],
        )
        self.decoder = DeepPerceiverDecoder(config=config.decoder_config)

        self.norm = nn.LayerNorm(
            config.decoder_config.perceiver_block_config.hidden_dim,
            eps=config.decoder_config.perceiver_block_config.eps,
        )

        assert config.linear_projection_config is not None
        self.prediction_layer = LinearProjection(config=config.linear_projection_config)

    def compute_rope_args(
        self,
        surface_position_batch_idx: torch.Tensor,
        surface_position: torch.Tensor,
        surface_position_supernode_idx: torch.Tensor,
        query_position: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute the RoPE frequency arguments for the surface_position and query_position. If we don't use RoPE,
        return empty dicts.
        """
        if not self.use_rope:
            return {}, {}

        # supernode pooling needs a sparse tensor. However, for the output after the supernode pooling,
        # we need to know the cast the shapes back to the batch dimension.
        batch_size = surface_position_batch_idx.unique().shape[0]
        supernode_freqs = self.rope(surface_position[surface_position_supernode_idx])
        channels = supernode_freqs.shape[-1]
        if supernode_freqs.ndim == 2:
            supernode_freqs = supernode_freqs.unsqueeze(0)  # add batch dimension
        # bring back the batch dimension
        supernode_freqs = supernode_freqs.reshape(batch_size, -1, channels)
        encoder_attn_kwargs = dict(freqs=supernode_freqs)
        decoder_attn_kwargs = dict(
            q_freqs=self.rope(query_position),
            k_freqs=supernode_freqs,
        )

        return encoder_attn_kwargs, decoder_attn_kwargs

    def forward(
        self,
        surface_position_batch_idx: torch.Tensor,
        surface_position_supernode_idx: torch.Tensor,
        surface_position: torch.Tensor,
        query_position: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the UPT model.

        Args:
            surface_position_batch_idx: Batch indices for the surface positions, since the surface positions are
                a sparse tensor for the supernode pooling.
            surface_position_supernode_idx: Supernode indices for the surface positions.
            surface_position: Input coordinates of the surface points.
            query_position: Input coordinates of the query points.

        Returns:
            torch.Tensor: Output tensor containing the predictions for the surface and volume fields, sliced according to the data specifications.
        """

        encoder_attn_kwargs, decoder_attn_kwargs = self.compute_rope_args(
            surface_position_batch_idx, surface_position, surface_position_supernode_idx, query_position
        )

        # supernode pooling encoder
        x = self.encoder(
            input_pos=surface_position,
            supernode_idx=surface_position_supernode_idx,
            batch_idx=surface_position_batch_idx,
        )
        # approximator blocks
        for block in self.approximator_blocks:
            x = block(x, attn_kwargs=encoder_attn_kwargs)

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
