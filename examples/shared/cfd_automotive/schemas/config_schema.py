#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pydantic import Field

from examples.shared.cfd_automotive.schemas.datasets import AeroDatasetConfig
from examples.shared.cfd_automotive.schemas.trainers.automotive_aerodynamics_trainer_config import (
    AutomotiveAerodynamicsCfdTrainerConfig,
)
from examples.tutorial.schemas.models.any_model_config import AnyModelConfig
from noether.core.schemas import ConfigSchema
from noether.core.schemas.dataset import AeroDataSpecs
from noether.core.schemas.statistics import AeroStatsSchema


class AeroAutomotiveCFDConfigSchema(ConfigSchema):
    data_specs: AeroDataSpecs
    model: AnyModelConfig = Field(..., discriminator="name")
    trainer: AutomotiveAerodynamicsCfdTrainerConfig
    datasets: dict[str, AeroDatasetConfig]
    dataset_statistics: AeroStatsSchema | None = None
