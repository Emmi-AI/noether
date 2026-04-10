#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ModelSize(str, Enum):
    """Available AB-UPT model sizes."""

    small = "small"
    scaled = "scaled"
    scaled_mps = "scaled_mps"


ABUPT_MODEL_KIND = "noether.modeling.models.aerodynamics.AeroABUPT"
TRAINER_KIND = "noether.training.trainers.WeightedLossTrainer"

FIELD_WEIGHTS = {
    "surface_pressure": 1.0,
    "surface_friction": 1.0,
    "volume_pressure": 1.0,
    "volume_velocity": 1.0,
    "volume_vorticity": 1.0,
}


@dataclass
class ABUPTSizeConfig:
    """AB-UPT model and pipeline configuration for a given size tier."""

    model_params: dict[str, Any]
    pipeline_overrides: dict[str, Any]


_PHYSICS_BLOCKS_SHORT: list[str] = ["perceiver", "self", "cross", "self", "cross", "self"]
_PHYSICS_BLOCKS_LONG: list[str] = ["perceiver"] + ["self", "cross"] * 5

MODEL_SIZES: dict[str, ABUPTSizeConfig] = {
    "small": ABUPTSizeConfig(
        model_params=dict(
            hidden_dim=192,
            num_heads=3,
            geometry_depth=1,
            physics_blocks=_PHYSICS_BLOCKS_SHORT,
            num_domain_decoder_blocks={"surface": 2, "volume": 2},
            radius=0.25,
        ),
        pipeline_overrides=dict(
            num_geometry_points=16384,
            num_geometry_supernodes=1024,
            num_surface_anchor_points=512,
            num_volume_anchor_points=512,
        ),
    ),
    "scaled": ABUPTSizeConfig(
        model_params=dict(
            hidden_dim=384,
            num_heads=6,
            geometry_depth=1,
            physics_blocks=_PHYSICS_BLOCKS_SHORT,
            num_domain_decoder_blocks={"surface": 6, "volume": 6},
            radius=0.1,
        ),
        pipeline_overrides=dict(
            num_geometry_points=125000,
            num_geometry_supernodes=32000,
            num_surface_anchor_points=16000,
            num_volume_anchor_points=32000,
        ),
    ),
    # Scaled model with reduced point budget for MPS (Apple Silicon)
    "scaled_mps": ABUPTSizeConfig(
        model_params=dict(
            hidden_dim=384,
            num_heads=6,
            geometry_depth=1,
            physics_blocks=_PHYSICS_BLOCKS_SHORT,
            num_domain_decoder_blocks={"surface": 6, "volume": 6},
            radius=0.1,
        ),
        pipeline_overrides=dict(
            num_geometry_points=16384,
            num_geometry_supernodes=4096,
            num_surface_anchor_points=2048,
            num_volume_anchor_points=4096,
        ),
    ),
}
