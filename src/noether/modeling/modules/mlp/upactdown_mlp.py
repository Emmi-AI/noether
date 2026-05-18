#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
from pydantic import BaseModel, Field, model_validator
from torch import nn

from noether.core.types import InitWeightsMode
from noether.modeling.functional.init import init_trunc_normal_zero_bias, init_xavier_uniform_zero_bias
from noether.modeling.modules.activations import Activation


class UpActDownMLPConfig(BaseModel):
    input_dim: int = Field(..., ge=1)
    """Input dimension of the MLP."""
    hidden_dim: int = Field(..., ge=2)
    """Hidden dimension of the MLP."""
    bias: bool = Field(True)
    """Whether to use bias in the MLP."""
    init_weights: InitWeightsMode = Field("truncnormal002")
    """ Initialization method of the weights of the MLP. Options are  "torch" (i.e., similar to the module) or  'truncnormal002'. Defaults to 'truncnormal002'."""

    @model_validator(mode="after")
    def check_dims(self) -> "UpActDownMLPConfig":
        """Validator to check that hidden_dim is greater than input_dim.

        Raises:
            ValueError: raised if hidden_dim is not greater than input_dim.
        """

        if not self.hidden_dim > self.input_dim:
            raise ValueError("hidden_dim should be greater than input_dim, otherwise it is not an up-projection")
        return self


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
        self.act = Activation.GELU.build()
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
        elif self.init_weights == "xavier":
            self.apply(init_xavier_uniform_zero_bias)
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
        x = self.act(x)
        x = self.fc2(x)
        return x
