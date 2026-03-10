#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pydantic import BaseModel

from noether.core.schemas.dataset import AeroDataSpecs
from noether.core.schemas.statistics import AeroStatsSchema


class AeroCFDPipelineConfig(BaseModel):
    kind: str
    num_surface_points: int
    """Number of surface points we sample as input for the model. """
    num_volume_points: int
    """Number of volume points we sample as input for the model."""
    num_surface_queries: int | None = None
    """ Number of surface queries we use to query the output function. Defaults to 0. If set to 0, no query points are sampled."""
    num_volume_queries: int | None = None
    """ Number of volume queries we use to query the output function. Defaults to 0. If set to 0, no query points are sampled."""
    use_physics_features: bool = False
    """ Whether to use physics features next to input coordinates (i.e., SDF and normal vectors). Defaults to False."""
    dataset_statistics: AeroStatsSchema | None = None
    """Dataset statistics (mean, std, max, min, etc) for normalization of input features."""
    sample_query_points: bool = True
    """Whether to sample query points. Defaults to True. If False, the query points are simply duplicated from the surface and volume points that serve as inputs for the encoder. This only applies for models that can query (e.g., UPT)."""
    num_supernodes: int = 0
    """ Number of supernodes (for UPT). """
    num_geometry_supernodes: int | None = None
    """ Number of geometry supernodes (for AB-UPT). """
    num_geometry_points: int | None = None
    """ Number of geometry points to sample (for AB-UPT). """
    num_volume_anchor_points: int | None = 0
    """ Number of volume anchor points to sample for AB-UPT. Defaults to 0."""
    num_surface_anchor_points: int | None = 0
    """Number of surface anchor points to sample for AB-UPT. Defaults to 0."""
    seed: int | None = None
    """Random seed for for processes that involve sampling (e.g., point sampling). Defaults to None."""
    data_specs: AeroDataSpecs
    """Data specifications for the pipeline."""
