#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Any

from aero_cfd.callbacks import AeroMetricsCallbackConfig
from noether.core.schemas.dataset import DomainDataSpec, ModelDataSpecs, RepeatWrapperConfig
from noether.core.schemas.normalizers import FieldNormalizerConfig
from noether.core.schemas.schema import ConfigSchema

from .base import AeroCFDPreset, AeroPipelineParams


class ShapeNetCarPreset(AeroCFDPreset):
    """Preset for the ShapeNet Car CFD dataset."""

    dataset_kind = "noether.data.datasets.cfd.ShapeNetCarDataset"

    pipeline_defaults: AeroPipelineParams = {
        "num_surface_points": 3586,
        "num_volume_points": 4096,
        "num_surface_queries": 3586,
        "num_volume_queries": 4096,
        "sample_query_points": False,
        "use_physics_features": False,
    }

    pipeline_model_overrides: dict[str, AeroPipelineParams] = {
        "noether.modeling.models.aerodynamics.AeroUPT": {
            "num_supernodes": 3586,
        },
        "noether.modeling.models.aerodynamics.AeroABUPT": {
            "num_geometry_supernodes": 512,
            "num_geometry_points": 3586,
            "num_surface_anchor_points": 256,
            "num_volume_anchor_points": 256,
            "num_surface_queries": 0,
            "num_volume_queries": 0,
        },
    }

    @property
    def data_specs(self) -> ModelDataSpecs:
        return ModelDataSpecs(
            position_dim=3,
            domains={
                "surface": DomainDataSpec(
                    output_dims={"pressure": 1},
                    feature_dim={"surface_sdf": 1, "surface_normals": 3},
                ),
                "volume": DomainDataSpec(
                    output_dims={"velocity": 3},
                    feature_dim={"volume_sdf": 1, "volume_normals": 3},
                ),
            },
        )

    @property
    def normalizer_spec(self) -> dict[str, FieldNormalizerConfig]:
        return {
            "surface_pressure": FieldNormalizerConfig(strategy="mean_std"),
            "volume_velocity": FieldNormalizerConfig(strategy="mean_std"),
            "volume_sdf": FieldNormalizerConfig(strategy="mean_std"),
            "surface_position": FieldNormalizerConfig(
                strategy="position", scale=1000, stat_keys={"min": "raw_pos_min", "max": "raw_pos_max"}
            ),
            "volume_position": FieldNormalizerConfig(
                strategy="position", scale=1000, stat_keys={"min": "raw_pos_min", "max": "raw_pos_max"}
            ),
        }

    @property
    def excluded_properties(self) -> set[str]:
        return {"surface_friction", "volume_pressure", "volume_vorticity"}

    def target_properties(self) -> list[str]:
        return ["surface_pressure_target", "volume_velocity_target"]

    def evaluation_callbacks(self, model_kind: str, *, batch_size: int = 1, every_n_epochs: int = 1) -> list:
        """Domain-specific evaluation callbacks for surface/volume metrics."""
        return [
            AeroMetricsCallbackConfig(
                batch_size=batch_size,
                every_n_epochs=every_n_epochs,
                dataset_key="test",
                forward_properties=self.forward_properties(model_kind),
            ),
        ]

    def build_config(
        self,
        *,
        model_kind: str,
        dataset_root: str,
        include_evaluation: bool = True,
        **kwargs: Any,
    ) -> ConfigSchema:
        """Build config with optional domain-specific evaluation callbacks and test_repeat dataset."""
        batch_size = kwargs.get("batch_size", 1)

        extra_callbacks = kwargs.pop("extra_callbacks", None) or []
        extra_datasets = kwargs.pop("extra_datasets", None) or {}

        if include_evaluation:
            extra_callbacks = self.evaluation_callbacks(model_kind, batch_size=batch_size) + extra_callbacks
            if "test_repeat" not in extra_datasets:
                extra_datasets["test_repeat"] = self.build_dataset(
                    split="test",
                    root=dataset_root,
                    model_kind=model_kind,
                    wrappers=[RepeatWrapperConfig(kind="noether.data.base.wrappers.RepeatWrapper", repetitions=10)],
                )

        return super().build_config(
            model_kind=model_kind,
            dataset_root=dataset_root,
            extra_callbacks=extra_callbacks,
            extra_datasets=extra_datasets,
            **kwargs,
        )
