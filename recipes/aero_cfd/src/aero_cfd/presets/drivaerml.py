#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from aero_cfd.callbacks.aero_metrics import AeroMetricsCallbackConfig
from noether.core.schemas.schema import ConfigSchema
from noether.data.preprocessors.normalizers import FieldNormalizerConfig
from noether.data.schemas import DomainDataSpec, ModelDataSpecs

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

    def evaluation_callbacks(
        self, model_kind: str, *, every_n_epochs: int = 1, max_epochs: int = 1, chunk_size: int = 1
    ) -> list:
        """Domain-specific evaluation callbacks for surface/volume metrics."""
        return [
            AeroMetricsCallbackConfig(
                batch_size=1,
                every_n_epochs=every_n_epochs,
                dataset_key="test",
                forward_properties=self.forward_properties(model_kind),
            ),
            AeroMetricsCallbackConfig(
                batch_size=1,
                every_n_epochs=max_epochs,  # Run at the end of training
                dataset_key="chunked_test",
                forward_properties=self.forward_properties(model_kind),
                chunked_inference=True,
                chunk_properties=["surface_anchor_position", "volume_anchor_position"],
                sample_size_property="surface_anchor_position",
                chunk_size=chunk_size,
            ),
        ]

    def build_config(
        self,
        *,
        model_kind: str,
        dataset_root: str,
        include_evaluation: bool = True,
        **kwargs,
    ) -> ConfigSchema:
        """Build config with optional domain-specific evaluation callbacks and test_repeat dataset."""
        max_epochs = kwargs.get("max_epochs", 1)
        chunk_size = kwargs.get("chunk_size", 1)

        extra_callbacks = kwargs.pop("extra_callbacks", None) or []
        extra_datasets = kwargs.pop("extra_datasets", None) or {}

        if include_evaluation:
            extra_callbacks = (
                self.evaluation_callbacks(model_kind, max_epochs=max_epochs, chunk_size=chunk_size) + extra_callbacks
            )

        return super().build_config(
            model_kind=model_kind,
            dataset_root=dataset_root,
            extra_callbacks=extra_callbacks,
            extra_datasets=extra_datasets,
            **kwargs,
        )

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
