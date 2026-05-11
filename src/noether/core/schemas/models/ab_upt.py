#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import warnings
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, computed_field, model_validator

from noether.core.schemas.dataset import FieldDimSpec, ModelDataSpecs
from noether.core.schemas.mixins import InjectSharedFieldFromParentMixin, Shared
from noether.core.schemas.modules.blocks import PerceiverBlockConfig, TransformerBlockConfig
from noether.core.schemas.modules.encoders import SupernodePoolingConfig
from noether.core.schemas.modules.layers import (
    ContinuousSincosEmbeddingConfig,
    LinearProjectionConfig,
    RopeFrequencyConfig,
)
from noether.core.schemas.modules.layers.vectors_conditioner import VectorsConditionerConfig
from noether.core.schemas.modules.mlp import MLPConfig
from noether.core.types import InitWeightsMode

from .base import ModelBaseConfig


class AnchorBranchedUPTConfig(ModelBaseConfig, InjectSharedFieldFromParentMixin):
    """Configuration for the Anchored Branched UPT (AB-UPT) model.

    AB-UPT is built from three configurable stages:

    1. **Geometry encoder** (optional): a :class:`SupernodePooling` encoder followed by
       ``geometry_depth`` standard transformer blocks. Only instantiated when at least
       one ``perceiver`` / ``perceiver_untied`` block is present in ``physics_blocks``
       and ``supernode_pooling_config`` is provided.
    2. **Physics trunk**: a stack of blocks listed in ``physics_blocks`` operating on
       per-domain anchor (and optionally query) tokens. The block string controls the
       attention pattern and weight sharing — see ``physics_blocks`` below.
    3. **Per-domain decoder** (optional): ``num_domain_decoder_blocks[name]``
       self-attention blocks with **untied weights per domain**, followed by a linear
       projection to that domain's output fields.

    ``hidden_dim`` is a shared field — it is auto-injected into
    ``transformer_block_config`` and ``supernode_pooling_config`` via
    :class:`InjectSharedFieldFromParentMixin`, so it only needs to be set once at the top
    level. See :doc:`/reference/config_inheritance`.

    Configuration guide
    -------------------

    See :doc:`/guides/training/configuring_ab_upt` for a step-by-step walkthrough of how
    to compose physics blocks, choose between tied and ``_untied`` variants, and wire up
    the per-domain decoder.

    Concrete examples (YAML):

    - Aerodynamics (multi-domain, surface + volume):
      `recipes/aero_cfd/configs/model/ab_upt.yaml
      <https://github.com/Emmi-AI/noether/blob/main/recipes/aero_cfd/configs/model/ab_upt.yaml>`_
    - Heat transfer (single-domain, volume only with parameter conditioning):
      `recipes/heat_transfer/configs/model/ab_upt.yaml
      <https://github.com/Emmi-AI/noether/blob/main/recipes/heat_transfer/configs/model/ab_upt.yaml>`_
    """

    kind: str | None = "noether.core.schemas.models.AnchorBranchedUPTConfig"

    model_config = ConfigDict(extra="forbid")

    supernode_pooling_config: Annotated[SupernodePoolingConfig, Shared] | None = None

    transformer_block_config: Annotated[TransformerBlockConfig, Shared]

    geometry_depth: int = Field(..., ge=0)
    """Number of transformer blocks in the geometry encoder."""

    hidden_dim: int = Field(..., ge=1)
    """Hidden dimension of the model."""

    condition_dim: int | None = Field(None, ge=1)

    physics_blocks: list[
        Literal[
            "self",
            "shared",
            "cross",
            "joint",
            "perceiver",
            "self_untied",
            "cross_untied",
            "joint_untied",
            "perceiver_untied",
        ]
    ]
    """Types of physics blocks to use in the model.

    self/shared: Self-attention within a branch/domain. Weights are shared between all domains.
    cross: Cross-attention between domains. Each domain attends to all other domains' anchors, weights are shared.
    joint: Joint attention over all domain points. Full self-attention over all points, weights are shared.
    perceiver: Perceiver-style cross-attention to geometry encoding.
    self_untied: Self-attention within a branch with untied weights for each domain.
    cross_untied: Cross-attention between domains with untied weights for each domain.
    joint_untied: Joint attention over all domain points with untied weights for each domain.
    perceiver_untied: Perceiver cross-attention with geometry encoding and untied weights per domain.

    Note: "shared" is a deprecated alias for "self" and will be removed in a future release."""

    num_domain_decoder_blocks: dict[str, int] = Field(default_factory=dict)
    """Number of final domain-specific decoder blocks with self attention and no weight sharing, e.g. {"surface": 2, "volume": 2}."""

    init_weights: InitWeightsMode = Field("truncnormal002")
    """Weight initialization of linear layers. Defaults to "truncnormal002"."""

    drop_path_rate: float = Field(0.0)
    """Drop path rate for stochastic depth. Defaults to 0.0 (no drop path)."""

    geometry_conditioning_dims: FieldDimSpec | None = None
    """Per-named-field conditioning spec for geometry transformer blocks. When
    left unset, defaults to ``data_specs.conditioning_dims`` so the geometry
    branch sees the same conditioning as the rest of the model. An explicit
    empty :class:`FieldDimSpec` (``total_dim == 0``) opts out — useful for
    diffusion, where timestep modulation should touch physics + per-domain
    decoders but not the geometry branch (geometry is invariant to noise level)."""

    data_specs: ModelDataSpecs
    """Data specifications for the model."""

    @model_validator(mode="after")
    def migrate_shared_to_self(self) -> "AnchorBranchedUPTConfig":
        """Migrate deprecated 'shared' block type to 'self'."""
        if "shared" in self.physics_blocks:
            warnings.warn(
                'physics_blocks: "shared" is deprecated, use "self" instead. '
                '"shared" will be removed in a future release.',
                DeprecationWarning,
                stacklevel=2,
            )
            self.physics_blocks = ["self" if b == "shared" else b for b in self.physics_blocks]
        return self

    @computed_field
    def rope_frequency_config(self) -> RopeFrequencyConfig:
        return RopeFrequencyConfig(
            hidden_dim=self.transformer_block_config.hidden_dim // self.transformer_block_config.num_heads,
            input_dim=self.data_specs.position_dim,
            implementation="complex",
            max_wavelength=self.transformer_block_config.max_wavelength,
        )

    @computed_field
    def pos_embed_config(self) -> ContinuousSincosEmbeddingConfig:
        return ContinuousSincosEmbeddingConfig(
            hidden_dim=self.hidden_dim,
            input_dim=self.data_specs.position_dim,
            max_wavelength=self.transformer_block_config.max_wavelength,
        )

    @computed_field
    def bias_mlp_config(self) -> MLPConfig:
        return MLPConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            bias=self.transformer_block_config.bias,
        )

    @computed_field
    def perceiver_block_config(self) -> PerceiverBlockConfig:
        return PerceiverBlockConfig(
            hidden_dim=self.hidden_dim,
            num_heads=self.transformer_block_config.num_heads,
            mlp_expansion_factor=self.transformer_block_config.mlp_expansion_factor,
            kv_dim=None,
            use_rope=self.transformer_block_config.use_rope,
            condition_dim=self.transformer_block_config.condition_dim,
            bias=self.transformer_block_config.bias,
        )

    @computed_field
    def domain_decoder_configs(self) -> dict[str, LinearProjectionConfig]:
        """Per-domain decoder projection configs, keyed by domain name."""
        return {
            name: LinearProjectionConfig(
                input_dim=self.hidden_dim,
                output_dim=spec.output_dims.total_dim,
                init_weights="truncnormal002",
                bias=self.transformer_block_config.bias,
            )
            for name, spec in self.data_specs.domains.items()
        }

    @computed_field
    def conditioner_config(self) -> VectorsConditionerConfig:
        """Configuration for the scalar conditioner module."""
        if self.data_specs.conditioning_dims is None or self.data_specs.conditioning_dims.total_dim == 0:
            raise ValueError(
                "Conditioning dims must be specified in data_specs and have total_dim > 0 to use the conditioner."
            )
        return VectorsConditionerConfig(
            hidden_dim=self.condition_dim,
            conditioning_spec=self.data_specs.conditioning_dims,
            condition_dim=self.condition_dim,  # type: ignore
            init_weights="truncnormal002",
        )

    @model_validator(mode="after")
    def set_condition_dim(self) -> "AnchorBranchedUPTConfig":
        """Set condition_dim in transformer_block_config based on data_specs."""

        if self.data_specs.conditioning_dims is not None and self.data_specs.conditioning_dims.total_dim > 0:
            if self.condition_dim is None:
                self.condition_dim = self.hidden_dim
            self.transformer_block_config.condition_dim = self.condition_dim

        if self.geometry_conditioning_dims is None and self.data_specs.conditioning_dims is not None:
            self.geometry_conditioning_dims = self.data_specs.conditioning_dims.model_copy(deep=True)

        return self

    @computed_field
    def geometry_transformer_block_config(self) -> TransformerBlockConfig:
        """Transformer block config for geometry encoder, with condition_dim set to geometry_conditioning_dims."""
        geo_block_config = self.transformer_block_config.model_copy(deep=True)
        if self.condition_dim is None:
            self.condition_dim = self.hidden_dim
        geo_block_config.condition_dim = self.condition_dim if self.geometry_conditioning_dims is not None else None  # type: ignore
        return geo_block_config

    @computed_field
    def geometry_conditioner_config(self) -> VectorsConditionerConfig:
        """Configuration for the scalar conditioner module."""
        if self.geometry_conditioning_dims is None or self.geometry_conditioning_dims.total_dim == 0:
            raise ValueError(
                "Geometry conditioning dims must be specified in data_specs and have total_dim > 0 to use the conditioner."
            )
        return VectorsConditionerConfig(
            hidden_dim=self.condition_dim,
            conditioning_spec=self.geometry_conditioning_dims,
            condition_dim=self.condition_dim,  # type: ignore
            init_weights="truncnormal002",
        )

    @model_validator(mode="after")
    def validate_parameters(self) -> "AnchorBranchedUPTConfig":
        """Validate validity of parameters across the model and its submodules.

        Ensures that hidden_dim is consistent across parent and all submodules.
        Note: transformer_block_config validates hidden_dim % num_heads == 0 in its own validator.
        """
        # SupernodePoolingConfig: hidden_dim equality
        if self.supernode_pooling_config is not None and self.supernode_pooling_config.hidden_dim != self.hidden_dim:
            raise ValueError(
                f"supernode_pooling_config.hidden_dim ({self.supernode_pooling_config.hidden_dim}) "
                f"must match model hidden_dim ({self.hidden_dim})."
            )

        # TransformerBlockConfig: hidden_dim equality
        if self.transformer_block_config.hidden_dim != self.hidden_dim:
            raise ValueError(
                f"transformer_block_config.hidden_dim ({self.transformer_block_config.hidden_dim}) "
                f"must match model hidden_dim ({self.hidden_dim})."
            )

        return self
