#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from pydantic import BaseModel, Field

from noether.core.types import InitWeightsMode


class ScalarsConditionerConfig(BaseModel):
    hidden_dim: int = Field(ge=1)
    """Dimension for embedding the scalars and the per-scalar MLP."""
    num_scalars: int = Field(ge=1)
    """How many scalars are embedded."""
    condition_dim: int | None = Field(None, ge=1)
    """Dimension of the final conditioning vector. Defaults to 4 * dim if condition_dim is None."""
    init_weights: InitWeightsMode = "truncnormal002"
    """Weight initialization for MLPs."""
