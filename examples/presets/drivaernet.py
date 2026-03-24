#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Any

from examples.presets.aero_cfd import AeroCFDPreset
from noether.core.schemas.dataset import AeroDataSpecs, DatasetBaseConfig, DatasetWrappers


class DrivAerNetPreset(AeroCFDPreset):
    """Preset for the DrivAerNet++ CFD dataset."""

    _dataset_kind = "noether.data.datasets.cfd.DrivAerNetDataset"

    _stats: dict[str, list[float]] = {
        "raw_pos_min": [-12.01],
        "raw_pos_max": [6.41],
        "surface_pressure_mean": [-9.34098e01],
        "surface_pressure_std": [1.20787e02],
        "surface_friction_mean": [-6.71649e-01, 3.63487e-02, -8.46379e-02],
        "surface_friction_std": [8.19941e-01, 4.51045e-01, 7.81055e-01],
        "volume_velocity_mean": [2.18719e01, -2.37778e-01, 6.73902e-01],
        "volume_velocity_std": [1.21079e01, 3.97768e00, 3.90113e00],
        "volume_pressure_mean": [-6.24053e01],
        "volume_pressure_std": [9.42394e01],
        "volume_vorticity_logscale_mean": [2.57623e-02, 2.58335e-01, 4.29835e-01],
        "volume_vorticity_logscale_std": [3.00179e00, 3.65020e00, 3.33356e00],
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
        from tutorial.schemas.datasets import AeroDatasetConfig

        return AeroDatasetConfig(
            kind=self._dataset_kind,
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
