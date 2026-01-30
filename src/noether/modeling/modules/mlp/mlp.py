#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
import torch.nn as nn

from noether.core.schemas.modules.mlp import MLPConfig
from noether.modeling.functional.init import init_trunc_normal_zero_bias
from noether.modeling.modules.activations import Activation


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
            config: Configuration object for the MLP.
        """
        super().__init__()

        # input layer and non-linearity
        layers = [nn.Linear(config.input_dim, config.hidden_dim), Activation[config.activation].value]
        self.init_weights = config.init_weights
        # hidden layers and non-linearities
        for _ in range(config.num_layers):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(Activation[config.activation].value)
        # output layer
        layers.append(nn.Linear(config.hidden_dim, config.output_dim))
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
