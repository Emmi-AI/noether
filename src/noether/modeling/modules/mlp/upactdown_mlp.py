#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
from torch import nn

from noether.core.schemas.modules.mlp import UpActDownMLPConfig
from noether.modeling.functional.init import init_trunc_normal_zero_bias
from noether.modeling.modules.activations import Activation


class UpActDownMlp(nn.Module):
    """UpActDownMlp is a vanilla MLP with an up-projection followed by an GELU activation function and a
    down-projection to the original input dim.
    """

    def __init__(
        self,
        config: UpActDownMLPConfig,
    ) -> None:
        """Initialize the UpActDownMlp.

        Args:
            config: The configuration of the UpActDownMlp.
        """

        super().__init__()

        self.init_weights = config.init_weights

        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim, bias=config.bias)
        self.act = Activation.RELU.build()
        self.fc2 = nn.Linear(config.hidden_dim, config.input_dim, bias=config.bias)

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
        elif self.init_weights == "truncnormal002-identity":
            self.apply(init_trunc_normal_zero_bias)
            nn.init.zeros_(self.fc2.weight)
        else:
            raise NotImplementedError(
                f"Initialization method {self.init_weights} not implemented. "
                "Use 'torch', 'truncnormal', or 'truncnormal002'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of the UpActDownMlp.

        Args:
            x: Input tensor to the MLP.

        Returns:
            Output tensor from the MLP.
        """

        x = self.fc1(x)
        x = self.act(x) ** 2
        x = self.fc2(x)
        return x
