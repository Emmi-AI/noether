#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
from torch import nn

from noether.modeling.models import Transformer as TransformerBackbone
from tutorial.schemas.models.transolver_config import TransolverConfig

from .base import BaseModel


class Transolver(BaseModel):
    """Implementation of the Transolver model.
    Reference code: https://github.com/thuml/Transolver/
    Paper: https://arxiv.org/abs/2402.02366

    Args:
        BaseModel: Base model class that contains the utilities for all models we use in this tutorial.
    """

    def __init__(
        self,
        model_config: TransolverConfig,
        **kwargs,
    ):
        """

        Args:
            model_config: Configuration of the Transolver model.
            attn_ctor: Attention constructor
        """

        super().__init__(model_config=model_config, **kwargs)

        # original implementation uses a weird dimension-wise scaling after embed (also not excluded from wd)
        # https://github.com/thuml/Transolver/blob/main/Car-Design-ShapeNetCar/models/Transolver.py#L163
        self.placeholder = nn.Parameter(torch.rand(1, 1, model_config.hidden_dim) / model_config.hidden_dim)

        self.transolver_backbone = TransformerBackbone(
            config=model_config
        )  # Transolver is a Transformer with a different attention mechanism

    def forward(
        self,
        surface_position: torch.Tensor,
        volume_position: torch.Tensor,
        surface_features: torch.Tensor | None = None,
        volume_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """forward pass of the Transolver model.

        Args:
            surface_position: input coordinates of the surface points.
            volume_position: input coordinates of the volume points.
            surface_mask_input: mask for the input points, indicating which points are surface points.
            physics_features: physics features for the input points. Defaults to None.
        Returns:
            dict[str, torch.Tensor]: dictionary with the output tensors, containing the surface pressure and volume velocity.
        """
        surface_mask_input = torch.zeros(
            surface_position.shape[0], surface_position.shape[1] + volume_position.shape[1]
        )
        surface_mask_input[:, : surface_position.shape[1]] = 1.0
        input_position = torch.concat([surface_position, volume_position], dim=1)
        x = self.pos_embed(input_position)

        if self.use_physics_features:
            surface_features = self.project_surface_features(surface_features)
            volume_features = self.project_volume_features(volume_features)
            physics_features = torch.concat([surface_features, volume_features], dim=1)
            x = x + physics_features

        x = self.surface_and_volume_bias(x=x, surface_mask=surface_mask_input)

        x = x + self.placeholder

        x = self.transolver_backbone(x=x, attn_kwargs={})

        x = self.output_projection(x)

        return self.gather_outputs(x=x, surface_mask=surface_mask_input)
