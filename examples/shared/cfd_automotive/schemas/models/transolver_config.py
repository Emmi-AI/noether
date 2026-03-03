#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from noether.core.schemas.models import TransolverConfig

from .base_config import TutorialBaseModelConfig


class TransolverConfig(TutorialBaseModelConfig, TransolverConfig):
    """expansion factor for the MLP."""

    name: Literal["transolver"] = "transolver"
