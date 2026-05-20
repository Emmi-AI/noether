#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import torch
from pydantic import BaseModel, Field
from torch import nn

from noether.core.types import InitWeightsMode
from noether.data.schemas import FieldDimSpec
from noether.modeling.modules.activations import Activation
from noether.modeling.modules.layers import LinearProjection, LinearProjectionConfig
from noether.modeling.modules.layers.continuous_sincos_embed import (
    ContinuousSincosEmbed,
    ContinuousSincosEmbeddingConfig,
)


class VectorsConditionerConfig(BaseModel):
    """Configuration for :class:`VectorsConditioner`.

    All conditioning inputs are expected to be normalized to ``[-1, 1]``; the
    underlying sine-cosine embedding runs in NeRF mode.
    """

    hidden_dim: int = Field(ge=1)
    """Dimension of the per-vector embedding and per-vector MLP."""
    conditioning_spec: FieldDimSpec
    """Mapping from input vector name to its feature dimension, e.g.
    ``{"angle": 1, "shape_params": 3}``."""

    condition_dim: int | None = Field(None, ge=1)
    """Dimension of the final conditioning vector. Defaults to ``hidden_dim`` if ``None``."""
    max_frequency: float = Field(1024.0, ge=1.0)
    """Highest frequency band, in units of ``π``, for the NeRF-mode sine-cosine
    embedding. Pick based on the smallest spatial scale you need to resolve in
    normalized coordinates (rough heuristic: ``1 / typical_input_spacing``)."""
    init_weights: InitWeightsMode = "truncnormal002"
    """Weight initialization for MLPs."""


class VectorsConditioner(nn.Module):
    def __init__(
        self,
        config: VectorsConditionerConfig,
    ):
        """Embeds a set of named vectors into a single conditioning vector.

        Each input vector named in ``config.conditioning_spec`` is encoded with a
        NeRF-mode :class:`ContinuousSincosEmbed` followed by a per-vector MLP. The
        resulting per-vector embeddings are concatenated and projected to
        ``condition_dim`` by a shared MLP.

        .. note::

            All input vectors must be normalized to ``[-1, 1]``. The underlying
            sine-cosine embedding uses NeRF-style frequencies tuned for that range;
            values outside it will alias and produce uninformative embeddings.

        Args:
            config: configuration for the VectorsConditioner. See :class:`~noether.core.schemas.modules.layers.vectors_conditioner.VectorsConditionerConfig` for available options.
        """
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.condition_dim = config.condition_dim
        self.conditioning_spec = config.conditioning_spec
        self.embedder = nn.ModuleDict(
            {
                name: nn.Sequential(
                    ContinuousSincosEmbed(
                        ContinuousSincosEmbeddingConfig(
                            hidden_dim=config.hidden_dim,
                            input_dim=dims,
                            mode="nerf",
                            max_frequency=config.max_frequency,
                        )
                    ),
                    LinearProjection(
                        LinearProjectionConfig(
                            input_dim=config.hidden_dim,
                            output_dim=config.hidden_dim,
                            init_weights=config.init_weights,  # type: ignore[arg-type]
                        )  # type: ignore[call-arg]
                    ),
                    Activation.GELU.build(),
                )
                for name, dims in self.conditioning_spec.items()
            },
        )
        # combine conditions
        self.shared_mlp = nn.Sequential(
            LinearProjection(
                LinearProjectionConfig(
                    input_dim=config.hidden_dim * len(self.conditioning_spec),
                    output_dim=config.hidden_dim,
                    init_weights=config.init_weights,  # type: ignore[arg-type]
                )  # type: ignore[call-arg]
            ),
            Activation.GELU.build(),
            LinearProjection(
                LinearProjectionConfig(
                    input_dim=config.hidden_dim,
                    output_dim=self.condition_dim,
                    init_weights=config.init_weights,  # type: ignore[arg-type]
                )  # type: ignore[call-arg]
            ),
            Activation.GELU.build(),
        )

    def forward(self, **conditioning_inputs: torch.Tensor) -> torch.Tensor:
        """Embed a set of named vectors into a single conditioning vector.

        All vectors declared in ``config.conditioning_spec`` must be supplied as
        keyword arguments matching the spec names. Inputs must be normalized to
        ``[-1, 1]``.

        Args:
            **conditioning_inputs: Vectors with shape ``(batch_size, num_features)``,
                keyed by the names declared in ``config.conditioning_spec``. The
                ``num_features`` of each vector must match the dimension declared in
                the spec. All inputs must share the same ``batch_size``.

        Returns:
            Conditioning vector with shape ``(batch_size, condition_dim)``.

        Raises:
            ValueError: If the supplied inputs don't match the spec (wrong number of
                vectors, missing key, wrong rank, or wrong feature dimension).

        Example:

            .. code-block:: python

                conditioner = VectorsConditioner(
                    VectorsConditionerConfig(
                        hidden_dim=64,
                        conditioning_spec={"angle": 1, "shape_params": 3},
                        condition_dim=128,
                        max_frequency=1024,
                    )
                )
                # Inputs normalized to [-1, 1].
                angle = torch.tensor([[0.5], [-0.2]])  # shape (batch_size, 1)
                shape_params = torch.tensor([[0.1, -0.3, 0.7], [-0.5, 0.2, -0.8]])  # shape (batch_size, 3)
                condition = conditioner(angle=angle, shape_params=shape_params)
                # condition.shape == (2, 128)
        """
        if len(conditioning_inputs) != len(self.conditioning_spec):
            raise ValueError(f"got {len(conditioning_inputs)} vectors but expected {len(self.conditioning_spec)}")

        for name, spec in self.conditioning_spec.items():
            if name not in conditioning_inputs:
                raise ValueError(f"missing vector {name} in input")
            vector = conditioning_inputs[name]
            if vector.shape[-1] != spec:
                raise ValueError(f"vector {name} should have {spec} features, got {vector.shape[1]}")

        projs = [self.embedder[name](conditioning_inputs[name]) for name in self.conditioning_spec.keys()]
        embed: torch.Tensor = self.shared_mlp(torch.concat(projs, dim=-1))

        return embed
