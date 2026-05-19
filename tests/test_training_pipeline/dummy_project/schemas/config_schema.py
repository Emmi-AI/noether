#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schemas.schema import ConfigSchema

from .datasets.base_dataset_config import BaseDatasetConfig
from .models.base_model_config import BaseModelConfig
from .trainer.base_trainer_config import BaseTrainerConfig


class BoilerplateConfigSchema(ConfigSchema):
    model: BaseModelConfig
    trainer: BaseTrainerConfig
    datasets: dict[str, BaseDatasetConfig]
