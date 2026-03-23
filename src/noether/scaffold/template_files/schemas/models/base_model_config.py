#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schemas import ModelBaseConfig


class BaseModelConfig(ModelBaseConfig):
    hidden_dim: int
    bias: bool = True
    num_hidden_layers: int = 0
    activation_function: str = "gelu"
    use_skip_connections: bool = False
    dropout: float = 0.0
    norm_layer: str | None = None
    input_dim: int
    output_dim: int
