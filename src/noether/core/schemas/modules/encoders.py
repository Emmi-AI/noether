#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from noether.core.types import InitWeightsMode


class SupernodePoolingConfig(BaseModel):
    hidden_dim: int = Field(..., ge=1)
    """Hidden dimension for positional embeddings, messages and the resulting output vector."""
    input_dim: int = Field(..., ge=1)
    """Number of positional dimension (e.g., input_dim=2 for a 2D position, input_dim=3 for a 3D position)"""
    radius: float | None = Field(None, ge=0.0)
    """Radius around each supernode. From points within this radius, messages are passed to the supernode."""
    k: int | None = Field(None, ge=1)
    """Number of neighbors for each supernode. From the k-NN points, messages are passed to the supernode."""
    max_degree: int = Field(32, ge=1)
    """Maximum degree of the radius graph. Defaults to 32."""
    spool_pos_mode: Literal["abspos", "relpos", "absrelpos"] = Field("abspos")
    """Type of position embedding: absolute space ("abspos"), relative space ("relpos") or both ("absrelpos")."""
    init_weights: InitWeightsMode = Field("truncnormal002")
    """ Weight initialization of linear layers. Defaults to "truncnormal002"."""
    readd_supernode_pos: bool = Field(True)
    """If true, the absolute positional encoding of the supernode is concatenated to the supernode vector after message passing and linearly projected back to hidden_dim. Defaults to True."""
    aggregation: Literal["mean", "sum"] = Field("mean")
    """Aggregation for message passing ("mean" or "sum")."""
    message_mode: Literal["mlp", "linear", "identity"] = Field("mlp")
    """How messages are created. "mlp" (2 layer MLP), "linear" (nn.Linear), "identity" (nn.Identity). Defaults to "mlp"."""
    input_features_dim: int | None = Field(None, ge=0)
    """Number of input features per point. None will fall back to a version without features. Defaults to None, which means no input features."""

    @model_validator(mode="after")
    def validate_radius_and_k(self):
        if (self.radius is None) and (self.k is None):
            raise ValueError("Either radius or k has to be set.")
        if (self.radius is not None) and (self.k is not None):
            raise ValueError("Only one of radius or k can be set.")
        return self
