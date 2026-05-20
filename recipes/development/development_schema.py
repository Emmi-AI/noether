#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from development.dataset import DevelopmentDatasetConfig
from development.model import DevelopmentModelConfig
from development.trainer import DevelopmentTrainerConfig
from pydantic import Field

from noether.core.schemas.schema import ConfigSchema


class DevelopmentSchema(ConfigSchema):
    development_batch_size: int
    datasets: dict[str, DevelopmentDatasetConfig] = Field(...)
    output_path: str | None = Field(default=None)  # set to None
    trainer: DevelopmentTrainerConfig | None = Field(default=None)  # Placeholder for trainer configuration
    model: DevelopmentModelConfig | None = Field(default=None)
