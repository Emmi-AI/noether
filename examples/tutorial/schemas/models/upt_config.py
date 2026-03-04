#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from examples.shared.cfd_automotive.schemas.models.base_config import AeroAutomotivCFDBaseModelConfig
from noether.core.schemas.models import UPTConfig


class UPTConfig(AeroAutomotivCFDBaseModelConfig, UPTConfig):
    name: Literal["upt"] = "upt"
