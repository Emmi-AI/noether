#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections import OrderedDict

from pydantic import BaseModel, Field, RootModel, model_validator


class FieldDimSpec(RootModel[OrderedDict[str, int]]):
    """A specification for a group of named data fields and their dimensions."""

    @property
    def field_slices(self) -> dict[str, slice]:
        """Calculates slice indices for each field in concatenation order."""
        indices = {}
        start = 0
        for field, dim in self.root.items():
            if not isinstance(dim, int) or dim <= 0:
                continue
            indices[field] = slice(start, start + dim)
            start += dim
        return indices

    @property
    def total_dim(self) -> int:
        """Calculates the total dimension of all fields combined."""
        return sum(self.root.values())

    def __getitem__(self, key: str) -> int:
        return self.root[key]

    def __iter__(self):
        return iter(self.root.items())

    def __getattr__(self, name: str) -> int:
        """Enables attribute-style access (e.g., `spec.geometry`)."""
        try:
            return self.root[name]
        except KeyError as err:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from err

    def __dir__(self) -> list[str]:
        """Improves autocompletion for dynamic attributes."""
        return sorted(set(super().__dir__()) | set(self.root.keys()))

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()

    def items(self):
        return self.root.items()

    def __len__(self):
        return len(self.root)


class DomainDataSpec(BaseModel):
    """Data specification for a single domain (e.g., surface, volume, wake)."""

    output_dims: FieldDimSpec
    """Output fields and their dimensions for this domain, e.g. {"pressure": 1, "velocity": 3}."""
    feature_dim: FieldDimSpec | None = None
    """Input feature fields and their dimensions for this domain."""


class ModelDataSpecs(BaseModel):
    """Base data specification for models that operate on arbitrary named domains.

    This is the minimal interface that model configs need from data specifications:
    position dimensions, available conditioning, and per-domain data descriptions.
    """

    position_dim: int = Field(..., ge=1)
    """Dimension of the input position vectors."""
    conditioning_dims: FieldDimSpec | None = None
    """Available conditioning features and their dimensions."""
    domains: dict[str, DomainDataSpec] = Field(default_factory=dict)
    """Per-domain data specifications keyed by domain name."""
    use_physics_features: bool = True
    """Whether physics features are used as input."""

    @property
    def total_output_dim(self) -> int:
        """Calculates the total output dimension across all domains."""
        return sum(spec.output_dims.total_dim for spec in self.domains.values())

    @property
    def all_targets(self) -> set[str]:
        """Returns all target field names across all domains, prefixed by domain name."""
        targets: set[str] = set()
        for name, spec in self.domains.items():
            targets |= {f"{name}_{key}" for key in spec.output_dims.keys()}
        return targets

    @property
    def all_features(self) -> set[str]:
        """Returns all feature field names across all domains."""
        features: set[str] = set()
        for spec in self.domains.values():
            if spec.feature_dim:
                features |= set(spec.feature_dim.keys())
        return features

    @model_validator(mode="after")
    def remove_feature_fields(self):
        if not self.use_physics_features:
            for spec in self.domains.values():
                spec.feature_dim = None
        return self


class FileMap(BaseModel):
    """File mapping schema for aerodynamic datasets.

    Maps field names to their corresponding file names in the dataset directory.
    This allows different datasets to use different file naming conventions while maintaining a unified interface.
    """

    # Surface field files
    surface_position: str | None = None
    surface_pressure: str | None = None
    surface_friction: str | None = None
    surface_normals: str | None = None
    surface_area: str | None = None

    # Volume field files
    volume_position: str | None = None
    volume_pressure: str | None = None
    volume_velocity: str | None = None
    volume_vorticity: str | None = None
    volume_normals: str | None = None

    # Optional additional surface position files (dataset-specific)
    surface_position_stl: str | None = None
    surface_position_stl_resampled: str | None = None

    # Optional volume friction
    volume_friction: str | None = None

    # Optional volume distance field
    volume_distance_to_surface: str | None = None

    # Optional design parameters file
    design_parameters: str | None = None


#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence

from pydantic import BaseModel


class AeroStatsSchema(BaseModel):
    """Unified statistics dataclass for aerodynamics datasets such as AhmedML, and DrivAerML, DrivAerNet++,
    ShapeNet-Car, and Wing."""

    # Surface statistics
    surface_domain_min: tuple[float, float, float] | None = None
    surface_domain_max: tuple[float, float, float] | None = None
    surface_pos_mean: tuple[float, float, float] | None = None
    surface_pos_std: tuple[float, float, float] | None = None
    surface_pressure_mean: tuple[float] | None = None
    surface_pressure_std: tuple[float] | None = None
    surface_friction_mean: tuple[float, float, float] | None = None
    surface_friction_std: tuple[float, float, float] | None = None

    # Volume statistics
    volume_pos_mean: tuple[float, float, float] | None = None
    volume_pos_std: tuple[float, float, float] | None = None
    volume_pressure_mean: tuple[float] | None = None
    volume_pressure_std: tuple[float] | None = None
    volume_velocity_mean: tuple[float, float, float] | None = None
    volume_velocity_std: tuple[float, float, float] | None = None
    volume_vorticity_mean: tuple[float, float, float] | None = None
    volume_vorticity_std: tuple[float, float, float] | None = None
    volume_vorticity_logscale_mean: tuple[float, float, float] | None = None
    volume_vorticity_logscale_std: tuple[float, float, float] | None = None
    volume_vorticity_magnitude_mean: float | None = None
    volume_vorticity_magnitude_std: float | None = None
    volume_domain_min: tuple[float, float, float] | None = None
    volume_domain_max: tuple[float, float, float] | None = None
    volume_sdf_mean: tuple[float] | None = None
    volume_sdf_std: tuple[float] | None = None

    # Inflow design parameter statistics
    inflow_design_parameters_min: Sequence[float] | None = None
    inflow_design_parameters_max: Sequence[float] | None = None
    inflow_design_parameters_mean: Sequence[float] | None = None
    inflow_design_parameters_std: Sequence[float] | None = None

    # Geometry design parameter statistics
    geometry_design_parameters_min: Sequence[float] | None = None
    geometry_design_parameters_max: Sequence[float] | None = None
    geometry_design_parameters_mean: Sequence[float] | None = None
    geometry_design_parameters_std: Sequence[float] | None = None

    # raw position statistics
    raw_pos_min: tuple[float] | None = None
    raw_pos_max: tuple[float] | None = None
