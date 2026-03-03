#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from noether.core.schemas.models import TransolverPlusPlusConfig

from .base_config import TutorialBaseModelConfig


class TransolverPlusPlusConfig(TutorialBaseModelConfig, TransolverPlusPlusConfig):
    name: Literal["transolver_plusplus"] = "transolver_plusplus"
