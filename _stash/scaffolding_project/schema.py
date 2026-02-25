#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from noether.core.schemas import ConfigSchema

from .dataset import DatasetConfig
from .model import ModelConfig
from .trainer import TrainerConfig


class ConfigSchema(ConfigSchema):
    # Custom config options can be added here
    model: ModelConfig
    trainer: TrainerConfig
    datasets: dict[str, DatasetConfig]
