#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pydantic import BaseModel, Field


class CompletePConfig(BaseModel):
    """Configuration for the CompleteP parameterization.

    CompleteP scales optimizer hyperparameters and model components based on width and depth
    multipliers relative to a base model, enabling hyperparameter transfer across scales.

    Reference: "Don't be lazy: CompleteP enables compute-efficient deep transformers" (NeurIPS 2025).
    """

    base_width: int = Field(256, ge=1)
    """Base width (hidden_dim) for computing the width multiplier m_w = hidden_dim / base_width."""

    base_depth: int = Field(2, ge=1)
    """Base depth (num_layers) for computing the depth multiplier m_d = depth / base_depth."""

    depth_alpha_exp: float = Field(1.0, ge=0.5, le=1.0)
    """Depth scaling exponent (alpha). Controls how residual branches are scaled: x = x + m_d^(-alpha) * branch(x).
    alpha=1.0 is CompleteP (complete feature learning), alpha=0.5 is the minimal stable parameterization."""

    init_std: float = Field(0.02, gt=0.0)
    """Base standard deviation for weight initialization. Hidden weights use init_std / sqrt(m_w)."""

    output_alpha: float = Field(1.0, gt=0.0)
    """Tunable multiplier for output scaling. Output is scaled by output_alpha / m_w."""
