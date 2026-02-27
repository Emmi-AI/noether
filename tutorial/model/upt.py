#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch

from noether.modeling.models import UPT as UPTBackbone
from tutorial.schemas.models import UPTConfig

from .base import BaseModel


class UPT(BaseModel):
    """Implementation of the UPT (Universal Physics Transformer) model.

    Args:
        BaseModel: Base model class that contains the utilities for all models we use in this tutorial.
    """

    def __init__(
        self,
        model_config: UPTConfig,
        **kwargs,
    ):
        """
        Args:
           model_config: Configuration of the tutorial UPT model, extending the core UPTConfig with tutorial-specific parameters.
        """
        super().__init__(model_config=model_config, **kwargs)

        self.upt_backbone = UPTBackbone(
            config=model_config,
        )

        self.use_bias_layers = model_config.use_bias_layers
        self.use_physics_features = model_config.data_specs.use_physics_features

    def forward(
        self,
        surface_position_batch_idx: torch.Tensor,
        surface_position_supernode_idx: torch.Tensor,
        surface_position: torch.Tensor,
        surface_query_position: torch.Tensor,
        volume_query_position: torch.Tensor,
        surface_features: torch.Tensor | None = None,
        surface_query_features: torch.Tensor | None = None,
        volume_query_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the UPT model.

        Args:
            surface_position_batch_idx: Batch indices for the surface positions, since the surface positions are a sparse tensor for the supernode pooling.
            surface_position_supernode_idx: Supernode indices for the surface positions.
            surface_position: Input coordinates of the surface points.
            surface_query_position: Surface query positions.
            volume_query_position: Volume query positions.
            surface_features: Surface features for the input points. Defaults to None.
            surface_query_features: Surface features for the query points. Defaults to None.
            volume_query_features: Volume features for the query points. Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Dictionary with the output tensors, containing the surface pressure and volume velocity.
        """

        # add features to queries
        if surface_features is None:
            surface_input_features = None
        else:
            surface_input_features = surface_features.squeeze(
                0
            )  # remove batch dimension, since we only have one sample
        query_position = torch.cat([surface_query_position, volume_query_position], dim=1)
        surface_mask_query = torch.zeros(surface_query_position.shape[0], query_position.shape[1])
        surface_mask_query[:, : surface_query_position.shape[1]] = 1.0

        encoder_attn_kwargs, decoder_attn_kwargs = self.upt_backbone.compute_rope_args(
            surface_position_batch_idx, surface_position, surface_position_supernode_idx, query_position
        )

        # supernode pooling encoder
        x = self.upt_backbone.encoder(
            input_pos=surface_position,
            supernode_idx=surface_position_supernode_idx,
            batch_idx=surface_position_batch_idx,
            input_features=surface_input_features,
        )
        # approximator blocks
        for block in self.upt_backbone.approximator_blocks:
            x = block(x, attn_kwargs=encoder_attn_kwargs)

        queries = self.upt_backbone.pos_embed(query_position)

        if self.use_bias_layers:
            queries = self.surface_and_volume_bias(queries, surface_mask_query)
        if self.use_physics_features:
            surface_query_features = self.project_surface_features(surface_query_features)
            volume_query_features = self.project_volume_features(volume_query_features)
            physics_query_features = torch.cat([surface_query_features, volume_query_features], dim=1)
            queries = queries + physics_query_features

        # perceiver decoder
        x = self.upt_backbone.decoder(
            kv=x,
            queries=queries,
            attn_kwargs=decoder_attn_kwargs,
            condition=None,
        )

        x = self.upt_backbone.norm(x)
        x = self.upt_backbone.prediction_layer(x)

        return self.gather_outputs(
            x=x,
            surface_mask=surface_mask_query,
        )
