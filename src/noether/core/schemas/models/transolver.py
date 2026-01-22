#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from pydantic import ConfigDict

from noether.core.schemas.models.transformer import TransformerConfig

from .base import ModelBaseConfig


class TransolverConfig(TransformerConfig, ModelBaseConfig):
    """Configuration for a Transolver model."""

    model_config = ConfigDict(extra="forbid")

    attention_constructor: Literal["transolver", "transolver_plusplus"] = "transolver"

    attention_arguments: dict = {"num_slices": 512}  # test if this can be overwritten in the model config


class TransolverPlusPlusConfig(TransolverConfig):
    """Configuration for a Transolver++ model."""

    attention_constructor: Literal["transolver_plusplus"] = "transolver_plusplus"
