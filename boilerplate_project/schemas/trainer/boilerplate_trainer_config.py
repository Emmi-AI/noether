#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Union

from pydantic import Field

from boilerplate_project.schemas.callbacks.base_callback_config import BoilerplateCallbackConfig
from noether.core.schemas import BaseTrainerConfig, CallbacksConfig

AllCallbacks = Union[BoilerplateCallbackConfig | CallbacksConfig]


class BoilerPlateTrainerConfig(BaseTrainerConfig):
    input_dim: int
    callbacks: list[AllCallbacks] | None = Field(..., description="List of callback configurations")
