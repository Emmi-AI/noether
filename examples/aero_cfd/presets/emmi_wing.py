#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Any

from examples.aero_cfd.presets.base import AeroCFDPreset, AeroPipelineParams
from noether.core.schemas.dataset import AeroDataSpecs


class EmmiWingPreset(AeroCFDPreset):
    """Preset for the EMMI Wing CFD dataset."""

    dataset_kind = "noether.data.datasets.cfd.EmmiWingDataset"

    pipeline_defaults: AeroPipelineParams = {
        "num_surface_points": 16384,
        "num_volume_points": 16384,
        "num_surface_queries": 0,
        "num_volume_queries": 0,
        "use_physics_features": False,
    }

    pipeline_model_overrides: dict[str, AeroPipelineParams] = {
        "noether.modeling.models.aerodynamics.AeroUPT": {
            "num_supernodes": 16384,
            "sample_query_points": False,
            "num_surface_queries": 16384,
            "num_volume_queries": 16384,
        },
        "noether.modeling.models.aerodynamics.AeroABUPT": {
            "num_geometry_supernodes": 1024,
            "num_geometry_points": 16384,
            "num_surface_anchor_points": 512,
            "num_volume_anchor_points": 512,
            "num_surface_queries": 0,
            "num_volume_queries": 0,
        },
    }

    forward_properties_map: dict[str, list[str]] = {
        "noether.modeling.models.aerodynamics.AeroUPT": [
            "surface_position_batch_idx",
            "surface_position_supernode_idx",
            "surface_position",
            "surface_query_position",
            "volume_query_position",
        ],
        "noether.modeling.models.aerodynamics.AeroABUPT": [
            "geometry_position",
            "geometry_supernode_idx",
            "geometry_batch_idx",
            "surface_anchor_position",
            "volume_anchor_position",
            "geometry_design_parameters",
            "inflow_design_parameters",
        ],
        "_default": [
            "surface_position",
            "volume_position",
            "surface_features",
            "volume_features",
        ],
    }

    @property
    def data_specs(self) -> AeroDataSpecs:
        return AeroDataSpecs(
            position_dim=3,
            surface_output_dims={"pressure": 1, "friction": 3},
            volume_output_dims={"pressure": 1, "velocity": 3, "vorticity": 3},
            conditioning_dims={
                "geometry_design_parameters": 5,
                "inflow_design_parameters": 2,
            },
        )

    @property
    def normalizer_spec(self) -> dict[str, str | tuple[str, dict[str, Any]]]:
        return {
            "surface_pressure": "mean_std",
            "surface_friction": "mean_std",
            "volume_pressure": "mean_std",
            "volume_velocity": "mean_std",
            # Wing uses magnitude-based normalization for vorticity (mean=0, std=magnitude_mean)
            "volume_vorticity": (
                "mean_std",
                {
                    "mean_key": "_zero",
                    "std_key": "volume_vorticity_magnitude_mean",
                },
            ),
            "geometry_design_parameters": "mean_std",
            "inflow_design_parameters": "mean_std",
            "surface_position": ("position", {"scale": 1000}),
            "volume_position": ("position", {"scale": 1000}),
        }

    @property
    def excluded_properties(self) -> set[str]:
        return {"surface_normals", "volume_normals", "volume_sdf"}

    def target_properties(self) -> list[str]:
        return [
            "surface_pressure_target",
            "surface_friction_target",
            "volume_pressure_target",
            "volume_velocity_target",
            "volume_vorticity_target",
        ]
