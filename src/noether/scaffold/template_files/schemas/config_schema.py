#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __PROJECT__.schemas.datasets import AeroDatasetConfig
from pydantic import Field

from noether.core.schemas import ConfigSchema as ConfigSchemaBase
from noether.core.schemas.dataset import AeroDataSpecs
from noether.core.schemas.statistics import AeroStatsSchema

from .models.any_model_config import AnyModelConfig
from .trainers.automotive_aerodynamics_trainer_config import AutomotiveAerodynamicsCfdTrainerConfig


class ConfigSchema(ConfigSchemaBase):
    data_specs: AeroDataSpecs
    model: AnyModelConfig = Field(..., discriminator="name")
    trainer: AutomotiveAerodynamicsCfdTrainerConfig
    datasets: dict[str, AeroDatasetConfig]
    dataset_statistics: AeroStatsSchema | None = None
