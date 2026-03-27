#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Any, TypedDict

from noether.core.presets import DomainPreset
from noether.core.schemas.dataset import DatasetBaseConfig, DatasetWrappers

AERO_PIPELINE = "tutorial.pipeline.AeroMultistagePipeline"


class AeroPipelineParams(TypedDict, total=False):
    """Typed parameters for the aero CFD pipeline.

    All fields are optional (``total=False``) since presets only set a subset, and model-specific
    overrides only touch a few fields.
    """

    num_surface_points: int
    num_volume_points: int
    num_surface_queries: int
    num_volume_queries: int
    use_physics_features: bool
    sample_query_points: bool
    num_supernodes: int
    num_geometry_supernodes: int
    num_geometry_points: int
    num_volume_anchor_points: int
    num_surface_anchor_points: int
    seed: int


class AeroCFDPreset(DomainPreset):
    """Intermediate base for automotive/aerospace CFD presets.

    Provides shared forward-property mappings for UPT/AB-UPT wrappers and concrete ``build_pipeline``/``build_dataset``
    implementations that use ``AeroCFDPipelineConfig`` and ``AeroDatasetConfig``.

    Domain presets (AhmedML, ShapeNetCar, etc.) inherit from this class and only need to specify data-specific
    attributes (stats, data specs, normalizers).
    """

    pipeline_defaults: AeroPipelineParams = {}  # type: ignore[assignment]
    pipeline_model_overrides: dict[str, AeroPipelineParams] = {}  # type: ignore[assignment]

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
        ],
        "_default": [
            "surface_position",
            "volume_position",
            "surface_features",
            "volume_features",
        ],
    }

    def build_pipeline(self, model_kind: str, **overrides: Any) -> Any:
        """Build an AeroCFDPipelineConfig with merged parameters."""
        from noether.core.schemas.aero import AeroCFDPipelineConfig
        from noether.core.schemas.statistics import AeroStatsSchema

        params = super().build_pipeline(model_kind, **overrides)
        return AeroCFDPipelineConfig(
            kind=AERO_PIPELINE,
            dataset_statistics=AeroStatsSchema(**self.dataset_statistics),
            data_specs=self.data_specs,
            **params,
        )

    def build_dataset(
        self,
        *,
        split: str,
        root: str,
        model_kind: str,
        wrappers: list[DatasetWrappers] | None = None,
        **overrides: Any,
    ) -> DatasetBaseConfig:
        """Build an AeroDatasetConfig for the given split."""
        from noether.core.schemas.aero import AeroDatasetConfig

        return AeroDatasetConfig(
            kind=self.dataset_kind,
            root=root,
            split=split,
            pipeline=self.build_pipeline(model_kind, **overrides),
            dataset_normalizers=self.build_normalizers(),
            dataset_wrappers=wrappers,
            excluded_properties=self.excluded_properties,
        )
