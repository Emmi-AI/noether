#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from noether.core.schemas.dataset import AeroDataSpecs, PipelineConfig
from noether.core.schemas.statistics import AeroStatsSchema


class AeroCFDPipelineConfig(PipelineConfig):
    """Pipeline configuration for aerodynamic CFD datasets."""

    num_surface_points: int
    """Number of surface points to sample as input for the model."""
    num_volume_points: int
    """Number of volume points to sample as input for the model."""
    num_surface_queries: int | None = None
    """Number of surface queries for the output function. If 0 or None, no query points are sampled."""
    num_volume_queries: int | None = None
    """Number of volume queries for the output function. If 0 or None, no query points are sampled."""
    use_physics_features: bool = False
    """Whether to use physics features (SDF, normals) alongside input coordinates."""
    dataset_statistics: AeroStatsSchema | None = None
    """Dataset statistics for normalization of input features."""
    sample_query_points: bool = True
    """Whether to sample query points. If False, query points are duplicated from encoder inputs."""
    num_supernodes: int = 0
    """Number of supernodes (for UPT)."""
    num_geometry_supernodes: int | None = None
    """Number of geometry supernodes (for AB-UPT)."""
    num_geometry_points: int | None = None
    """Number of geometry points to sample (for AB-UPT)."""
    num_volume_anchor_points: int | None = 0
    """Number of volume anchor points to sample for AB-UPT."""
    num_surface_anchor_points: int | None = 0
    """Number of surface anchor points to sample for AB-UPT."""
    seed: int | None = None
    """Random seed for sampling processes."""
    data_specs: AeroDataSpecs
    """Data specifications for the pipeline."""
