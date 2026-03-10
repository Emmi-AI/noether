#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from noether.core.schemas.models import TransformerConfig

from .base_config import BaseModelConfig


class TransformerConfig(BaseModelConfig, TransformerConfig):
    name: Literal["transformer"] = "transformer"
