#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Any

from examples.presets.aero_cfd import AeroCFDPreset
from noether.core.schemas.dataset import AeroDataSpecs


class AhmedMLPreset(AeroCFDPreset):
    """Preset for the AhmedML CFD dataset (CAEML benchmark)."""

    _dataset_kind = "noether.data.datasets.cfd.AhmedMLDataset"

    # _stats: dict[str, list[float]] = {
    #     "raw_pos_min": [-4.0],
    #     "raw_pos_max": [6.0],
    #     "surface_pressure_mean": [-1.00952e-01],
    #     "surface_pressure_std": [1.88242e-01],
    #     "surface_friction_mean": [-1.52900e-03, 7.83792e-09, -5.82453e-05],
    #     "surface_friction_std": [1.17512e-03, 6.52266e-04, 7.13125e-04],
    #     "volume_velocity_mean": [8.74600e-01, 1.42877e-05, 7.76145e-03],
    #     "volume_velocity_std": [3.00305e-01, 1.14927e-01, 1.24698e-01],
    #     "volume_pressure_mean": [8.12013e-01],
    #     "volume_pressure_std": [3.67992e-01],
    #     "volume_vorticity_logscale_mean": [-1.45271e-04, 1.29314e-01, 1.29501e-05],
    #     "volume_vorticity_logscale_std": [1.11017e00, 1.96530e00, 1.77495e00],
    # }
    _stats_file: str = "tutorial/configs/dataset_statistics/ahmedml_stats.yaml"

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
