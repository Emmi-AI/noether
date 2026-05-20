#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from noether.core.types import InitWeightsMode
from noether.modeling.functional.init import init_trunc_normal_zero_bias
from noether.modeling.modules.activations import Activation


class MLPConfig(BaseModel):
    input_dim: int = Field(..., ge=1)
    """Input dimension of the MLP."""
    output_dim: int = Field(..., ge=1)
    """Output dimension of the MLP."""
    hidden_dim: int = Field(..., ge=1)
    """Hidden dimension for each layer."""
    num_layers: int = Field(0, ge=0)
    """Number of hidden layers in the MLP. If 0, the MLP is a two linear layer MLP from input_dim, hidden_dim, activation to output_dim."""
    activation: Literal["RELU", "GELU", "SIGMOID", "TANH", "LEAKY_RELU", "SOFTPLUS", "ELU", "SILU"] = "GELU"
    """Activation function to use between layers."""
    init_weights: InitWeightsMode = "truncnormal002"
    """Weight initialization method."""
    bias: bool = Field(True)
    """Whether to use bias in the linear layers."""


class MLP(nn.Module):
    """
    Implements a Multi-Layer Perceptron (MLP) with configurable number of layers, hidden dimension activation functions and weight initialization methods.
    Only one hidden dimension is supported for simplicity, i.e., all hidden layers have the same dimension.
    The MLP will always have one input layer and one output layer. When num_layers=0, the MLP is a two layer network with one non-linearity in between.
    When num_layers>=1, the MLP has additional hidden layers, etc.
    """

    def __init__(
        self,
        config: MLPConfig,
    ) -> None:
        """Initialize the MLP.

        Args:
            config: Configuration object for the MLP. See :class:`~noether.core.schemas.modules.mlp.MLPConfig` for available options.
        """
        super().__init__()

        # input layer and non-linearity
        layers = [
            nn.Linear(config.input_dim, config.hidden_dim, bias=config.bias),
            Activation[config.activation].build(),
        ]
        self.init_weights = config.init_weights
        # hidden layers and non-linearities
        for _ in range(config.num_layers):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.bias))
            layers.append(Activation[config.activation].build())
        # output layer
        layers.append(nn.Linear(config.hidden_dim, config.output_dim, bias=config.bias))
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the MLP with a specific initialization. Options are "torch" (i.e., default), or
            "truncnormal002".

        Raises:
            NotImplementedError: raised if the specified initialization is not implemented.
        """

        if self.init_weights == "torch":
            pass
        elif self.init_weights == "truncnormal002":
            self.apply(init_trunc_normal_zero_bias)
        else:
            raise NotImplementedError(
                f"Initialization method {self.init_weights} not implemented. Use 'torch', 'truncnormal', or 'truncnormal002'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of the MLP.

        Args:
            x: Input tensor to the MLP.

        Returns:
            Output tensor from the MLP.
        """
        return self.mlp(x)  # type: ignore[no-any-return]
