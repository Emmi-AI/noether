#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from noether.core.schemas.models import AnchorBranchedUPTConfig

from .base_config import TutorialBaseModelConfig


class ABUPTConfig(TutorialBaseModelConfig, AnchorBranchedUPTConfig):
    name: Literal["ab_upt"] = "ab_upt"
