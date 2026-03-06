#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
import torch.nn as nn

from boilerplate_project.schemas.models.base_model_config import BaseModelConfig
from noether.core.models import Model

ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
}

NORM_LAYERS = {
    "layer_norm": nn.LayerNorm,
    "batch_norm": nn.BatchNorm1d,
    "instance_norm": nn.InstanceNorm1d,
    "group_norm": nn.GroupNorm,
}


class BaseModel(Model):
    """A simple base (MLP) model for classification tasks."""

    def __init__(
        self,
        model_config: BaseModelConfig,
        **kwargs,
    ):
        """

        Args:
            input_dim: input dimension of the model.
            hidden_dim: hidden dimension of the model.
            output_dim: output dimension of the model.
            bias: whether to use bias in the linear layers. Defaults to True.
            num_hidden_layers: number of hidden layers in the model. Defaults to 0.
            activation_function: activation function to use. Defaults to "gelu".
        """
        super().__init__(model_config=model_config, **kwargs)

        self.input_projection = nn.Linear(model_config.input_dim, model_config.hidden_dim, bias=model_config.bias)
        self.output_projection = nn.Linear(model_config.hidden_dim, model_config.output_dim, bias=model_config.bias)
        if model_config.activation_function not in ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Unsupported activation function: {model_config.activation_function}. Supported: {list(ACTIVATION_FUNCTIONS.keys())}"
            )
        self.activation = ACTIVATION_FUNCTIONS[model_config.activation_function]()
        if model_config.norm_layer is not None and model_config.norm_layer not in NORM_LAYERS:
            raise ValueError(
                f"Unsupported norm layer: {model_config.norm_layer}. Supported: {list(NORM_LAYERS.keys())}"
            )
        self.norm_layer = NORM_LAYERS[model_config.norm_layer]() if model_config.norm_layer else nn.Identity()

        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(model_config.hidden_dim, model_config.hidden_dim, bias=model_config.bias),
                    self.norm_layer,
                    self.activation,
                    nn.Dropout(model_config.dropout),
                )
                for _ in range(model_config.num_hidden_layers)
            ]
        )
        self.use_skip_connections = model_config.use_skip_connections

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the base model.

        Args:
            x: torch.Tensor
                Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor
                Output tensor of shape (batch_size, output_dim).
        """
        x = self.activation(self.input_projection(x))
        for layer in self.hidden_layers:
            if self.use_skip_connections:
                x = x + self.activation(layer(x))
            else:
                x = self.activation(layer(x))
        x = self.output_projection(x)
        return x
