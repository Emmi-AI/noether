#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections import OrderedDict

from pydantic import Field, computed_field
from torch import Tensor

from noether.core.schemas.dataset import DomainDataSpec, FieldDimSpec, ModelDataSpecs
from noether.core.schemas.mixins import InjectSharedFieldFromParentMixin
from noether.core.schemas.models import AnchorBranchedUPTConfig
from noether.modeling.models.ab_upt import ModelKVCache
from noether.modeling.models.aerodynamics import AeroABUPT


class UQABUPTConfig(AnchorBranchedUPTConfig, InjectSharedFieldFromParentMixin):
    """Config for UQ-wrapped AB-UPT model with heteroscedastic output and anchor subsampling."""

    # --- UQ-specific fields ---
    enable_heteroscedastic: bool = True
    """Enable aleatoric uncertainty via heteroscedastic output"""
    num_anchor_subsamples: int = Field(10, ge=1)
    """Number of anchor subsamples for epistemic estimation"""
    anchor_subsample_ratio: float = Field(0.8, gt=0.0, le=1.0)
    """Fraction of anchors to keep in each subsample"""
    min_log_variance: float = Field(-10.0)
    """Min clamp for predicted log-variance"""
    max_log_variance: float = Field(10.0)
    """Max clamp for predicted log-variance"""

    @computed_field
    def effective_data_specs(self) -> ModelDataSpecs:
        """If heteroscedastic is enabled, double the output dims so the decoder produces mean + log_var."""
        if not self.enable_heteroscedastic:
            return self.data_specs

        doubled_domains: dict[str, DomainDataSpec] = {}
        for domain_name, domain_spec in self.data_specs.domains.items():
            doubled_output = FieldDimSpec(OrderedDict((k, v * 2) for k, v in domain_spec.output_dims.root.items()))
            doubled_domains[domain_name] = DomainDataSpec(
                output_dims=doubled_output,
                feature_dim=domain_spec.feature_dim,
            )

        return ModelDataSpecs(
            position_dim=self.data_specs.position_dim,
            conditioning_dims=self.data_specs.conditioning_dims,
            domains=doubled_domains,
            use_physics_features=self.data_specs.use_physics_features,
        )

    @computed_field
    def parent_config(self) -> AnchorBranchedUPTConfig:
        """Parent config for shared fields."""
        data_specs = self.effective_data_specs if self.enable_heteroscedastic else self.data_specs
        return AnchorBranchedUPTConfig(
            data_specs=data_specs,
            **self.model_dump(include=set(AnchorBranchedUPTConfig.model_fields), exclude={"data_specs"}),
        )


class UQAnchoredBranchedUPT(AeroABUPT):
    """AB-UPT model wrapped with uncertainty quantification.

    Provides two UQ mechanisms:
    - Aleatoric (heteroscedastic): decoder outputs mean + log-variance per field
    - Epistemic (anchor subsampling): multiple forward passes with random anchor subsets
    """

    def __init__(self, model_config: UQABUPTConfig, **kwargs):
        super().__init__(model_config=model_config.parent_config, **kwargs)

        self.enable_heteroscedastic = model_config.enable_heteroscedastic
        self.original_data_specs = model_config.data_specs  # non-doubled specs
        self.min_log_var = model_config.min_log_variance
        self.max_log_var = model_config.max_log_variance
        self.num_anchor_subsamples = model_config.num_anchor_subsamples
        self.anchor_subsample_ratio = model_config.anchor_subsample_ratio

    def _split_mean_logvar(self, predictions: dict[str, Tensor]) -> dict[str, Tensor]:
        """Split doubled predictions into mean and log-variance for each field.

        The backbone outputs fields with 2x the original dimension when heteroscedastic
        is enabled. For each field (e.g. 'surface_pressure' with shape (B, N, 2*D)),
        we produce 'surface_pressure_mean' (B, N, D) and 'surface_pressure_log_var' (B, N, D).
        """
        # Build lookup of original field dims across all domains
        original_dims: dict[str, int] = {}
        for domain_spec in self.original_data_specs.domains.values():
            for name, dim in domain_spec.output_dims.root.items():
                original_dims[name] = dim

        split_preds: dict[str, Tensor] = {}
        for key, tensor in predictions.items():
            # key format: "{prefix}_{field_name}" e.g. "surface_pressure", "query_volume_velocity"
            # Find which original field this corresponds to
            field_name = self._extract_field_name(key)
            if field_name in original_dims:
                d = original_dims[field_name]
                prefix = key[: -len(field_name) - 1]  # e.g. "surface" or "query_volume"
                split_preds[f"{prefix}_{field_name}_mean"] = tensor[..., :d]
                log_var = tensor[..., d:]
                split_preds[f"{prefix}_{field_name}_log_var"] = log_var.clamp(self.min_log_var, self.max_log_var)
            else:
                split_preds[key] = tensor

        return split_preds

    @staticmethod
    def _extract_field_name(key: str) -> str:
        """Extract field name from prediction key.

        Keys follow pattern: '{domain}_{field}' or 'query_{domain}_{field}'.
        Domain is 'surface' or 'volume'.
        """
        parts = key.split("_")
        if parts[0] == "query":
            # query_surface_pressure -> pressure, query_volume_velocity -> velocity
            return "_".join(parts[2:])
        # surface_pressure -> pressure, volume_velocity -> velocity
        return "_".join(parts[1:])

    def forward(
        self,
        geometry_position: Tensor | None = None,
        geometry_supernode_idx: Tensor | None = None,
        geometry_batch_idx: Tensor | None = None,
        surface_anchor_position: Tensor | None = None,
        volume_anchor_position: Tensor | None = None,
        geometry_design_parameters: Tensor | None = None,
        inflow_design_parameters: Tensor | None = None,
        query_surface_position: Tensor | None = None,
        query_volume_position: Tensor | None = None,
        kv_cache: ModelKVCache | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass with optional heteroscedastic output splitting.

        During training, this is the primary method called by the trainer.
        Returns a flat dict of predictions (with _mean and _log_var suffixes if heteroscedastic).
        """
        predictions = super().forward(
            geometry_position=geometry_position,
            geometry_supernode_idx=geometry_supernode_idx,
            geometry_batch_idx=geometry_batch_idx,
            surface_anchor_position=surface_anchor_position,
            volume_anchor_position=volume_anchor_position,
            geometry_design_parameters=geometry_design_parameters,
            inflow_design_parameters=inflow_design_parameters,
            query_surface_position=query_surface_position,
            query_volume_position=query_volume_position,
            kv_cache=kv_cache,
        )

        if self.enable_heteroscedastic:
            return self._split_mean_logvar(predictions)
        return predictions
