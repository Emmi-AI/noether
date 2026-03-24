#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Any

from examples.presets.aero_cfd import AeroCFDPreset
from noether.core.schemas.dataset import AeroDataSpecs


class DrivAerMLPreset(AeroCFDPreset):
    """Preset for the DrivAerML CFD dataset (CAEML benchmark)."""

    _dataset_kind = "noether.data.datasets.cfd.DrivAerMLDataset"

    _stats: dict[str, list[float]] = {
        "raw_pos_min": [-40.0],
        "raw_pos_max": [80.0],
        "surface_pressure_mean": [-2.29772e02],
        "surface_pressure_std": [2.69345e02],
        "surface_friction_mean": [-1.20054e00, 1.49358e-03, -7.20107e-02],
        "surface_friction_std": [2.07670e00, 1.35628e00, 1.11426e00],
        "volume_velocity_mean": [1.67909e01, -3.82238e-02, 4.07968e-01],
        "volume_velocity_std": [1.64115e01, 8.63614e00, 6.64996e00],
        "volume_pressure_mean": [1.71387e-01],
        "volume_pressure_std": [5.00826e-01],
        "volume_vorticity_logscale_mean": [-1.47814e-02, 7.87642e-01, 2.81023e-03],
        "volume_vorticity_logscale_std": [5.45681e00, 5.77081e00, 5.46175e00],
    }

    _pipeline_defaults: dict[str, Any] = {
        "num_surface_points": 16384,
        "num_volume_points": 16384,
        "num_surface_queries": 0,
        "num_volume_queries": 0,
        "use_physics_features": False,
    }

    _pipeline_model_overrides: dict[str, dict[str, Any]] = {
        "noether.modeling.models.wrappers.UPTWrapper": {
            "num_supernodes": 16384,
            "sample_query_points": False,
            "num_surface_queries": 16384,
            "num_volume_queries": 16384,
        },
        "noether.modeling.models.wrappers.ABUPTWrapper": {
            "num_geometry_supernodes": 1024,
            "num_geometry_points": 16384,
            "num_surface_anchor_points": 512,
            "num_volume_anchor_points": 512,
            "num_surface_queries": 0,
            "num_volume_queries": 0,
        },
    }

    @property
    def data_specs(self) -> AeroDataSpecs:
        return AeroDataSpecs(
            position_dim=3,
            surface_output_dims={"pressure": 1, "friction": 3},
            volume_output_dims={"pressure": 1, "velocity": 3, "vorticity": 3},
        )

    @property
    def normalizer_spec(self) -> dict[str, str | tuple[str, dict[str, Any]]]:
        return {
            "surface_pressure": "mean_std",
            "surface_friction": "mean_std",
            "volume_pressure": "mean_std",
            "volume_velocity": "mean_std",
            "volume_vorticity": (
                "mean_std",
                {
                    "mean_key": "volume_vorticity_logscale_mean",
                    "std_key": "volume_vorticity_logscale_std",
                    "logscale": True,
                },
            ),
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
