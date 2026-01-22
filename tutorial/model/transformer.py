#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
import torch

from noether.core.schemas.modules.layers import ContinuousSincosEmbeddingConfig, RopeFrequencyConfig
from noether.modeling.models import Transformer as TransformerBackbone
from noether.modeling.modules.layers import ContinuousSincosEmbed, RopeFrequency
from tutorial.schemas.models.transformer_config import TransformerConfig

from .base import BaseModel


class Transformer(BaseModel):
    """Implementation of a Transformer model.

    Args:
        BaseModel: Base model class that contains the utilities for all models we use in this tutorial.
    """

    def __init__(
        self,
        model_config: TransformerConfig,
        **kwargs,
    ):
        """
        Args:
            model_config: Configuration of the Transformer model.
        """
        super().__init__(model_config=model_config, **kwargs)

        self.encoder = ContinuousSincosEmbed(
            config=ContinuousSincosEmbeddingConfig(hidden_dim=model_config.hidden_dim, input_dim=3)
        )

        self.use_rope = model_config.use_rope
        self.rope = (
            RopeFrequency(
                config=RopeFrequencyConfig(hidden_dim=model_config.hidden_dim // model_config.num_heads, input_dim=3)
            )
            if self.use_rope
            else None
        )

        self.transfomer_backbone = TransformerBackbone(config=model_config)

    def forward(
        self,
        surface_position: torch.Tensor,
        volume_position: torch.Tensor,
        surface_features: torch.Tensor | None = None,
        volume_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the Transformer model.

        Args:
            surface_position: input coordinates of the surface points.
            volume_position: input coordinates of the volume points.
            surface_mask_input: surface mask for the input points, indicating which points are surface points.
            surface_features: optional input features of the surface points.
            volume_features: optional input features of the volume points.

        Returns:
            dict[str, torch.Tensor]: dictionary with the output tensors, containing the surface pressure and volume velocity.
        """
        surface_mask_input = torch.zeros(
            surface_position.shape[0], surface_position.shape[1] + volume_position.shape[1]
        )
        surface_mask_input[:, : surface_position.shape[1]] = 1.0
        input_position = torch.concat([surface_position, volume_position], dim=1)
        attn_kwargs = {}

        if self.use_rope:
            rope = self.rope(input_position)
            attn_kwargs["freqs"] = rope

        x = self.encoder(input_position)

        if self.use_physics_features:
            surface_features = self.project_surface_features(surface_features)
            volume_features = self.project_volume_features(volume_features)
            physics_features = torch.concat([surface_features, volume_features], dim=1)
            x = x + physics_features
        # this step is redudant if the SDF is part of the physics_features. Without SDF, this step is needed.
        x = self.surface_and_volume_bias(x=x, surface_mask=surface_mask_input)

        x = self.transfomer_backbone(x=x, attn_kwargs=attn_kwargs)

        x = self.output_projection(x)

        return self.gather_outputs(x=x, surface_mask=surface_mask_input)
