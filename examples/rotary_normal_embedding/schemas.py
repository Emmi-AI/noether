#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from pydantic import Field

from tutorial.schemas.config_schema import TutorialConfigSchema
from tutorial.schemas.models.ab_upt_config import ABUPTConfig


class AnchorBranchedUPTConfig(ABUPTConfig):
    use_normal_rope: bool = Field(False)
    """Enable Rotary Normal Embedding (RoNE): encodes relative surface normal orientation
    alongside relative position in the RoPE attention bias. Only affects surface and geometry
    tokens; volume tokens keep classic position-only RoPE."""

    use_surface_normal_features: bool = Field(False)
    """When True, surface anchor token features are initialised with a learnable normal embedding
    (nn.Linear(3, hidden_dim)) added to the position embedding.  This injects normal information
    into the value stream of the physics blocks, complementing RoNE which only affects Q/K."""

    normal_rope_dim_fraction: float = Field(0.25, gt=0.0, lt=1.0)
    """Fraction of head_dim allocated to normal orientation encoding. The remaining fraction
    is used for position encoding. Only used when use_normal_rope=True."""

    normal_rope_max_wavelength: float = Field(10.0, gt=0.0)
    """Max wavelength for normal RoPE frequencies. Normals live in [-1, 1], so this should be
    much smaller than the position max_wavelength (default 10000). Only used when use_normal_rope=True."""

    cross_attention_normal_mode: Literal["zeros", "position_only"] = Field("position_only")
    """Controls how normal RoPE frequencies are handled in CrossAnchorAttention physics blocks
    (surface ↔ volume cross-attention). Only relevant when use_normal_rope=True.

    "zeros": volume tokens use zero normals (identity rotation) while surface tokens use their
        actual normal frequencies. This creates an asymmetry in cross-attention — surface Q is
        rotated but volume K is not — breaking the RoPE relative-encoding guarantee and leaking
        absolute surface-normal information into the cross-attention logit.
    "position_only": both surface and volume tokens use position-only frequencies in cross-attention
        blocks. Normal frequencies are suppressed for both sides, so RoPE's relative-encoding
        property is preserved. Recommended (and default) when use_normal_rope=True.

    Self-attention (shared) and joint blocks are unaffected — they always use the full
    position+normal frequencies."""


class RoNEConfigSchema(TutorialConfigSchema):
    model: AnchorBranchedUPTConfig = Field(..., discriminator="name")
