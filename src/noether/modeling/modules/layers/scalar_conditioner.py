#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
from torch import nn

from noether.core.schemas.modules.layers import ContinuousSincosEmbeddingConfig, LinearProjectionConfig
from noether.core.schemas.modules.layers.scalar_conditioner import ScalarsConditionerConfig
from noether.modeling.modules.activations import Activation
from noether.modeling.modules.layers import ContinuousSincosEmbed, LinearProjection


class ScalarsConditioner(nn.Module):
    def __init__(
        self,
        config: ScalarsConditionerConfig,
    ):
        """Embeds num_scalars scalars into a single conditioning vector via first encoding every scalar with
        sine-cosine embeddings followed by a mlp (per scalar). These vectors are then concatenated and projected down
        to condition_dim with an MLP.

        Args:
            hidden_dim: Dimension for embedding the scalars and the per-scalar MLP.
            num_scalars: How many scalars are embedded.
            condition_dim: Dimension of the final conditioning vector. Defaults to 4 * dim if condition_dim is None.
            init_weights: Weight initialization for MLPs.
        """
        super().__init__()
        condition_dim = config.condition_dim or config.hidden_dim * 4
        self.hidden_dim = config.hidden_dim
        self.num_scalars = config.num_scalars
        self.condition_dim = condition_dim
        # sin-cos embedding of individual scalars
        embed_config = ContinuousSincosEmbeddingConfig(hidden_dim=config.hidden_dim, input_dim=1)  # type: ignore[call-arg]
        self.embed = ContinuousSincosEmbed(embed_config)
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    LinearProjection(
                        LinearProjectionConfig(
                            input_dim=config.hidden_dim,
                            output_dim=config.hidden_dim,
                            init_weights=config.init_weights,  # type: ignore[arg-type]
                        )  # type: ignore[call-arg]
                    ),
                    Activation.GELU.value,
                )
                for _ in range(config.num_scalars)
            ],
        )
        # combine conditions
        self.shared_mlp = nn.Sequential(
            LinearProjection(
                LinearProjectionConfig(
                    input_dim=config.hidden_dim * config.num_scalars,
                    output_dim=config.hidden_dim,
                    init_weights=config.init_weights,  # type: ignore[arg-type]
                )  # type: ignore[call-arg]
            ),
            Activation.GELU.value,
            LinearProjection(
                LinearProjectionConfig(
                    input_dim=config.hidden_dim,
                    output_dim=self.condition_dim,
                    init_weights=config.init_weights,  # type: ignore[arg-type]
                )  # type: ignore[call-arg]
            ),
            Activation.GELU.value,
        )

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        """Embeds scalars into a single conditioning vector. Scalars can be passed as *args or as **kwargs. It is
        recommended to use kwargs to avoid bugs that originate from passing scalars in a different order at two
        locations in the code. Recommended usage: condition = conditioner(geometry_angle=75.3, friction_angle=24.6)
        Args:
            *args: Scalars in tensor representation (batch_size,) or (batch_size, 1).
            **kwargs: Scalars in tensor representation (batch_size,) or (batch_size, 1).
        Returns:
            Conditioning vector with shape (batch_size, condition_dim)
        """
        # checks + preprocess
        scalars: list[torch.Tensor] = [*args] + list(kwargs.values())
        assert len(scalars) == self.num_scalars, f"got {len(scalars)} scalars but num_scalars == {self.num_scalars}"
        expected_len = None
        for i in range(self.num_scalars):
            scalar = scalars[i]
            if expected_len is None:
                expected_len = scalar.numel()
            assert scalar.numel() == len(scalar) and scalar.ndim <= 2, (
                f"scalar should be (batch_size,) or (batch_size, 1), got {scalar.shape}"
            )
            assert len(scalar) == expected_len, f"got scalars of different size ({len(scalar)} != {expected_len})"
            if scalar.ndim == 1:
                scalars[i] = scalar.unsqueeze(1)

        # embed all scalars at once
        embeds = self.embed(torch.concat(scalars)).chunk(self.num_scalars)
        # project embeds
        projs = [self.mlps[i](embeds[i]) for i in range(self.num_scalars)]
        # combine embeds
        embed: torch.Tensor = self.shared_mlp(torch.concat(projs, dim=1))

        return embed
