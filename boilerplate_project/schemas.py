#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schemas import ConfigSchema, ModelBaseConfig
from noether.core.schemas.dataset import DatasetBaseConfig
from noether.core.schemas.trainers import BaseTrainerConfig


class BaseDatasetConfig(DatasetBaseConfig):
    num_samples: int
    """Total number of samples to generate."""
    num_classes: int = 10
    """The number of distinct classes (clusters) to generate."""
    noise: float = 0.1
    """The standard deviation of the Gaussian noise added to the data."""
    radius: float = 1.0
    """The radius of the circle on which the cluster centers are placed."""


class BaseModelConfig(ModelBaseConfig):
    hidden_dim: int
    bias: bool = True
    num_hidden_layers: int = 0
    activation_function: str = "gelu"
    use_skip_connections: bool = False
    dropout: float = 0.0
    input_dim: int
    output_dim: int


class BoilerplateConfigSchema(ConfigSchema):
    """Typed config schema for the boilerplate project.

    Provides Pydantic validation for project-specific model and dataset fields.
    The ``input_dim`` field at the root level allows sharing the value between
    the model and trainer configs via Hydra interpolation (``${input_dim}``).
    """

    input_dim: int = 2
    model: BaseModelConfig
    trainer: BaseTrainerConfig
    datasets: dict[str, BaseDatasetConfig]
