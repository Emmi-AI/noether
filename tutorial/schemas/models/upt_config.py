#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from pydantic import model_validator

from noether.core.schemas.models import UPTConfig

from .base_config import TutorialBaseModelConfig


class UPTConfig(TutorialBaseModelConfig, UPTConfig):
    name: Literal["upt"] = "upt"

    @model_validator(mode="after")
    def update_supernode_pooling_config(self) -> "UPTConfig":
        """Set input_features_dim on supernode pooling from the surface (geometry) feature dim."""
        if self.data_specs.use_physics_features:
            surface_spec = self.data_specs.domains.get("surface")
            if surface_spec and surface_spec.feature_dim:
                self.supernode_pooling_config.input_features_dim = surface_spec.feature_dim.total_dim
        return self
