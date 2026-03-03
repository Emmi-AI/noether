#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from noether.core.schemas.models import UPTConfig

from .base_config import TutorialBaseModelConfig


class UPTConfig(TutorialBaseModelConfig, UPTConfig):
    name: Literal["upt"] = "upt"
