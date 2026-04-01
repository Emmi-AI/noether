#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from pydantic import Field

from development.dataset import DevelopmentDatasetConfig
from noether.core.schemas import ConfigSchema


class DevelopmentSchema(ConfigSchema):
    datasets: dict[str, DevelopmentDatasetConfig] = Field(...)
    output_path: str | None = Field(default=None)  # set to None
    trainer: None = Field(default=None)  # Placeholder for trainer configuration
    model: None = Field(default=None)  # Placeholder for model configuration
