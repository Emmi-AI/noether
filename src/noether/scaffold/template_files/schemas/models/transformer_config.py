#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from pydantic import computed_field

from noether.core.schemas.models import TransformerConfig
from noether.core.schemas.modules.layers.continuous_sincos_embedding import ContinuousSincosEmbeddingConfig
from noether.core.schemas.modules.layers.rope_frequency import RopeFrequencyConfig

from .base_config import BaseModelConfig


class TransformerConfig(BaseModelConfig, TransformerConfig):
    name: Literal["transformer"] = "transformer"

    @computed_field
    def pos_encoding_config(self) -> ContinuousSincosEmbeddingConfig:
        return ContinuousSincosEmbeddingConfig(hidden_dim=self.transformer_block_config.hidden_dim, input_dim=3)

    @computed_field
    def rope_frequency_config(self) -> RopeFrequencyConfig:
        return RopeFrequencyConfig(hidden_dim=self.hidden_dim // self.transformer_block_config.num_heads, input_dim=3)
