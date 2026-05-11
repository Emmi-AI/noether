#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from pydantic import BaseModel, Field

from noether.core.schemas.dataset import FieldDimSpec
from noether.core.types import InitWeightsMode


class VectorsConditionerConfig(BaseModel):
    """Configuration for :class:`VectorsConditioner`.

    All conditioning inputs are expected to be normalized to ``[-1, 1]``; the
    underlying sine-cosine embedding runs in NeRF mode.
    """

    hidden_dim: int = Field(ge=1)
    """Dimension of the per-vector embedding and per-vector MLP."""
    conditioning_spec: FieldDimSpec
    """Mapping from input vector name to its feature dimension, e.g.
    ``{"angle": 1, "shape_params": 3}``."""

    condition_dim: int | None = Field(None, ge=1)
    """Dimension of the final conditioning vector. Defaults to ``hidden_dim`` if ``None``."""
    max_frequency: float = Field(1024.0, ge=1.0)
    """Highest frequency band, in units of ``π``, for the NeRF-mode sine-cosine
    embedding. Pick based on the smallest spatial scale you need to resolve in
    normalized coordinates (rough heuristic: ``1 / typical_input_spacing``)."""
    init_weights: InitWeightsMode = "truncnormal002"
    """Weight initialization for MLPs."""
