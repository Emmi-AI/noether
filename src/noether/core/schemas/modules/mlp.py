#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from noether.core.types import InitWeightsMode


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
