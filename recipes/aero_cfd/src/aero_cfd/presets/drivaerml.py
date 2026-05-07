#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from noether.core.schemas.dataset import DomainDataSpec, ModelDataSpecs
from noether.core.schemas.normalizers import FieldNormalizerConfig

from .base import AeroCFDPreset, AeroPipelineParams


class DrivAerMLPreset(AeroCFDPreset):
    """Preset for the DrivAerML CFD dataset (CAEML benchmark)."""

    dataset_kind = "noether.data.datasets.cfd.DrivAerMLDataset"

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

    @property
    def data_specs(self) -> ModelDataSpecs:
        return ModelDataSpecs(
            position_dim=3,
            domains={
                "surface": DomainDataSpec(output_dims={"pressure": 1, "friction": 3}),
                "volume": DomainDataSpec(output_dims={"pressure": 1, "velocity": 3, "vorticity": 3}),
            },
        )

    @property
    def normalizer_spec(self) -> dict[str, FieldNormalizerConfig]:
        return {
            "surface_pressure": FieldNormalizerConfig(strategy="mean_std"),
            "surface_friction": FieldNormalizerConfig(strategy="mean_std"),
            "volume_pressure": FieldNormalizerConfig(strategy="mean_std"),
            "volume_velocity": FieldNormalizerConfig(strategy="mean_std"),
            "volume_vorticity": FieldNormalizerConfig(
                strategy="mean_std",
                logscale=True,
                stat_keys={"mean": "volume_vorticity_logscale_mean", "std": "volume_vorticity_logscale_std"},
            ),
            "surface_position": FieldNormalizerConfig(
                strategy="position", scale=1000, stat_keys={"min": "raw_pos_min", "max": "raw_pos_max"}
            ),
            "volume_position": FieldNormalizerConfig(
                strategy="position", scale=1000, stat_keys={"min": "raw_pos_min", "max": "raw_pos_max"}
            ),
        }

    @property
    def excluded_properties(self) -> set[str]:
        return {"surface_normals", "surface_area", "volume_normals", "volume_sdf", "surface_sdf"}

    def target_properties(self) -> list[str]:
        return [
            "surface_pressure_target",
            "surface_friction_target",
            "volume_pressure_target",
            "volume_velocity_target",
            "volume_vorticity_target",
        ]
