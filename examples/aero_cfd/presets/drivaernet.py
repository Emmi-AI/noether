#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Any

from examples.aero_cfd.presets.base import AeroCFDPreset, AeroPipelineParams
from noether.core.schemas.dataset import AeroDataSpecs, DatasetBaseConfig, DatasetWrappers


class DrivAerNetPreset(AeroCFDPreset):
    """Preset for the DrivAerNet++ CFD dataset."""

    dataset_kind = "noether.data.datasets.cfd.DrivAerNetDataset"

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

    def build_dataset(
        self,
        *,
        split: str,
        root: str,
        model_kind: str,
        wrappers: list[DatasetWrappers] | None = None,
        filter_categories: tuple[str, ...] | None = None,
        **overrides: Any,
    ) -> DatasetBaseConfig:
        """Build dataset config with optional category filtering.

        Args:
            filter_categories: optional tuple of DrivAerNet design categories to include
                (e.g., ``("F_S_WWS_WM", "N_S_WWS_WM")``). None loads all categories.
        """
        from noether.core.schemas.aero import AeroDatasetConfig

        return AeroDatasetConfig(
            kind=self.dataset_kind,
            root=root,
            split=split,
            pipeline=self.build_pipeline(model_kind, **overrides),
            dataset_normalizers=self.build_normalizers(),
            dataset_wrappers=wrappers,
            excluded_properties=self.excluded_properties,
            filter_categories=filter_categories,
        )

    def target_properties(self) -> list[str]:
        return [
            "surface_pressure_target",
            "surface_friction_target",
            "volume_pressure_target",
            "volume_velocity_target",
            "volume_vorticity_target",
        ]
