#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch

from noether.core.models import CompositeModel, Model
from noether.core.schemas.modules.blocks import TransformerBlockConfig
from noether.core.schemas.modules.layers import ContinuousSincosEmbeddingConfig, RopeFrequencyConfig
from noether.modeling.modules.layers import ContinuousSincosEmbed, RopeFrequency
from tutorial.model.composite_components import CompositeTransformerBlock
from tutorial.schemas.models.composite_transformer_config import (
    CompositeTransformerBlockConfig,
    CompositeTransformerConfig,
)


class CompositeTransformer(CompositeModel):
    """This is an example of a composite Transformer model, having two stacks of transformer blocks which can have different optimizers.
        Note that this is mainly an example for the sake of demonstration and development, and not necessarily a useful architecture.

    Args:


    """

    def __init__(self, model_config: CompositeTransformerConfig, **kwargs):
        """A composite model consisting of multiple transformer-based single models."""
        super().__init__(model_config=model_config, **kwargs)

        self.model_config = model_config
        self.use_rope = model_config.use_rope
        if self.use_rope:
            self.rope = RopeFrequency(
                config=RopeFrequencyConfig(hidden_dim=model_config.hidden_dim // model_config.num_heads, input_dim=3)
            )

        self.pos_embed = ContinuousSincosEmbed(
            config=ContinuousSincosEmbeddingConfig(hidden_dim=model_config.hidden_dim, input_dim=3)
        )

        # name of the first submodel is 'low_level_blocks', second is 'high_level_blocks'
        self.low_level_blocks = CompositeTransformerBlock(
            model_config=CompositeTransformerBlockConfig(
                name=model_config.low_level_blocks.name,
                transformer_config=TransformerBlockConfig(
                    hidden_dim=model_config.hidden_dim,
                    num_heads=model_config.num_heads,
                    mlp_hidden_dim=model_config.mlp_expansion_factor * model_config.hidden_dim,
                    use_rope=model_config.use_rope,
                ),
                depth=model_config.low_level_blocks.depth,
                optimizer_config=model_config.low_level_blocks.optimizer_config,
                use_rope=model_config.use_rope,
                projection_bias=True,
                is_frozen=model_config.low_level_blocks.is_frozen,
            ),
            **kwargs,
        )

        self.high_level_blocks = CompositeTransformerBlock(
            model_config=CompositeTransformerBlockConfig(
                name=model_config.high_level_blocks.name,
                transformer_config=TransformerBlockConfig(
                    hidden_dim=model_config.hidden_dim,
                    num_heads=model_config.num_heads,
                    mlp_hidden_dim=model_config.mlp_expansion_factor * model_config.hidden_dim,
                    use_rope=model_config.use_rope,
                ),
                depth=model_config.high_level_blocks.depth,
                optimizer_config=model_config.high_level_blocks.optimizer_config,
                use_rope=model_config.use_rope,
                use_output_projection=True,
                output_dim=model_config.data_specs.total_output_dim,
                is_frozen=model_config.high_level_blocks.is_frozen,
            ),
            **kwargs,
        )

        self.required_batch_modes = {"input_position", "surface_mask_input"}

    @property
    def submodels(self) -> dict[str, Model]:
        """Returns the submodels of the composite model."""
        return dict(
            low_level_blocks=self.low_level_blocks,
            high_level_blocks=self.high_level_blocks,
        )

    def _gather_outputs(self, x: torch.Tensor, surface_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Gathers the outputs from the final tensor x into a dictionary.

        Args:
            x: The final output tensor from the transformer blocks.

        """

        surface_mask = surface_mask[0]  # we assume the surface mask is the same for all samples in the batch

        return dict(
            surface_pressure=x[:, surface_mask.bool(), :1],
            volume_velocity=x[:, ~surface_mask.bool(), 1:4],
        )

    def forward(
        self,
        surface_position: torch.Tensor,
        volume_position: torch.Tensor,
        physics_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the composite transformer model.

        Args:
            surface_position: Tensor of shape (B, N_surface, 3) representing the positions of surface points.
            volume_position: Tensor of shape (B, N_volume, 3) representing the positions of volume points.
            physics_features: Optional tensor of shape (B, N, D_phys) representing additional physics features.
        Returns:
            dict[str, torch.Tensor]: dictionary with the output tensors, containing the surface pressure and volume velocity.
        """
        surface_mask_input = torch.zeros(
            surface_position.shape[0], surface_position.shape[1] + volume_position.shape[1]
        )
        surface_mask_input[:, : surface_position.shape[1]] = 1.0
        input_position = torch.concat([surface_position, volume_position], dim=1)

        attn_kwargs = {}
        # rope does not have trainable parameters, and hence it can be part of the composite forward. If a module has trainable parameters, it should be part of a single model.
        if self.use_rope:
            rope = self.rope(input_position)
            attn_kwargs["freqs"] = rope

        # Encode input positions
        x = self.low_level_blocks(
            x=self.pos_embed(input_position), attn_kwargs=attn_kwargs, surface_mask=surface_mask_input
        )
        x = self.high_level_blocks(x=x, attn_kwargs=attn_kwargs)

        return self._gather_outputs(x=x, surface_mask=surface_mask_input)
