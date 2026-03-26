#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pydantic import Field

from noether.core.schemas import ConfigSchema
from noether.core.schemas.dataset import AeroDataSpecs
from noether.core.schemas.statistics import AeroStatsSchema
from tutorial.schemas.datasets import AeroDatasetConfig

from .models.any_model_config import AnyModelConfig
from .trainers.automotive_aerodynamics_trainer_config import AutomotiveAerodynamicsCfdTrainerConfig


class DevelopmentConfigSchema(ConfigSchema):
    data_specs: AeroDataSpecs
    model: AnyModelConfig | None = Field(None)
    trainer: AutomotiveAerodynamicsCfdTrainerConfig | None = Field(None)
    datasets: dict[str, AeroDatasetConfig] = Field(...)
    dataset_statistics: AeroStatsSchema | None = None
