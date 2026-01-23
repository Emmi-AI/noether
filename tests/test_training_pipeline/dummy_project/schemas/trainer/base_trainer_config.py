#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Union

from dummy_project.schemas.callbacks.base_callback_config import BoilerplateCallbackConfig
from pydantic import Field

from noether.core.schemas import BaseTrainerConfig
from noether.core.schemas.callbacks import CallbacksConfig

AllCallbacks = Union[BoilerplateCallbackConfig | CallbacksConfig]  #


class BaseTrainerConfig(BaseTrainerConfig):
    input_dim: int
    callbacks: list[AllCallbacks] | None = Field(..., description="List of callback configurations")
