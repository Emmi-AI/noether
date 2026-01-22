#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch
import torch.nn as nn

from noether.core.models import Model
from noether.core.providers import PathProvider
from noether.core.schemas.modules.layers import LinearProjectionConfig
from noether.core.utils.training.counter import UpdateCounter
from noether.data.container import DataContainer
from noether.modeling.modules.blocks import TransformerBlock
from noether.modeling.modules.layers import LinearProjection
from tutorial.schemas.models.composite_transformer_config import CompositeTransformerBlockConfig


class CompositeTransformerBlock(Model):
    def __init__(
        self,
        model_config: CompositeTransformerBlockConfig,
        update_counter: UpdateCounter | None = None,
        path_provider: PathProvider | None = None,
        data_container: DataContainer | None = None,
        static_context: dict[str, Any] | None = None,
    ):
        """Composite Transformer Block Model.

        Args:
            model_config: The model configuration used to initialize the model.
            is_frozen: If true, will set `requires_grad` of all parameters to false. Will also put the model into eval
                mode (e.g., to put Dropout or BatchNorm into eval mode).
            path_provider: A path provider used by the initializer to store or retrieve checkpoints.
            data_container: The data container which includes the data and dataloader.
                This is currently unused but helpful for quick prototyping only, evaluating forward in debug mode, etc.
            static_context: The static context used to pass information between submodules, e.g. patch_size, latent_dim.
        """
        super().__init__(
            model_config=model_config,
            is_frozen=model_config.is_frozen,
            update_counter=update_counter,
            path_provider=path_provider,
            data_container=data_container,
            static_context=static_context,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config=model_config.transformer_config,
                )
                for _ in range(model_config.depth)
            ]
        )
        self.output_projection = None
        if model_config.use_output_projection:
            self.norm = nn.LayerNorm(model_config.transformer_config.hidden_dim, eps=1e-6)
            self.output_projection = LinearProjection(
                config=LinearProjectionConfig(
                    input_dim=model_config.transformer_config.hidden_dim,
                    output_dim=model_config.output_dim,
                    intit_weights="truncnormal002",
                )
            )

        if model_config.projection_bias:
            self.surface_bias = nn.Sequential(
                LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=model_config.transformer_config.hidden_dim,
                        output_dim=model_config.transformer_config.hidden_dim,
                        init_weights="truncnormal002",
                    )
                ),
                nn.GELU(),
                LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=model_config.transformer_config.hidden_dim,
                        output_dim=model_config.transformer_config.hidden_dim,
                        init_weights="truncnormal002",
                    )
                ),
            )

            self.volume_bias = nn.Sequential(
                LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=model_config.transformer_config.hidden_dim,
                        output_dim=model_config.transformer_config.hidden_dim,
                        init_weights="truncnormal002",
                    )
                ),
                nn.GELU(),
                LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=model_config.transformer_config.hidden_dim,
                        output_dim=model_config.transformer_config.hidden_dim,
                        init_weights="truncnormal002",
                    )
                ),
            )

    def surface_and_volume_bias(self, x: torch.Tensor, surface_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies separate biases to surface and volume points based on the surface mask.

        """
        unbatch = False
        if x.ndim == 2:
            # if we have a single point, we need to add a batch dimension
            unbatch = True
            x = x.unsqueeze(0)

        surface_mask = surface_mask[0]  #
        x_surface = self.surface_bias(x[:, surface_mask.bool(), :])
        x_volume = self.volume_bias(x[:, ~surface_mask.bool(), :])
        x = torch.concat([x_surface, x_volume], dim=1)
        if unbatch:
            x = x.squeeze(0)
        return x

    def forward(
        self,
        x: torch.Tensor,
        attn_kwargs: dict[str, torch.Tensor] | None = None,
        surface_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the Composite Transformer Block Model.

        Args:
            x: Input tensor of shape (B, N, D_in) or (N, D_in) where B is the batch size, N is the number of points,
                and D_in is the input feature dimension.
            attn_kwargs: Optional dictionary containing additional arguments for attention mechanisms.
            surface_mask: Optional tensor indicating which points belong to the surface (used if projection_bias is True).

        Returns:
            A dictionary containing the output tensors.
        """

        if self.model_config.projection_bias:
            x = self.surface_and_volume_bias(x, surface_mask=surface_mask)

        for block in self.blocks:
            x = block(x, attn_kwargs=attn_kwargs)

        if self.output_projection is not None:
            x = self.norm(x)
            x = self.output_projection(x)

        return x
