#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pydantic import Field

from noether.core.schemas import ConfigSchema
from noether.core.schemas.aero import AeroDatasetConfig
from noether.core.schemas.dataset import ModelDataSpecs
from noether.core.schemas.statistics import AeroStatsSchema

from .models.any_model_config import AnyModelConfig
from .trainers.automotive_aerodynamics_trainer_config import AutomotiveAerodynamicsCfdTrainerConfig


class TutorialConfigSchema(ConfigSchema):
    data_specs: ModelDataSpecs
    model: AnyModelConfig = Field(..., discriminator="name")
    trainer: AutomotiveAerodynamicsCfdTrainerConfig
    datasets: dict[str, AeroDatasetConfig]
    dataset_statistics: AeroStatsSchema | None = None
