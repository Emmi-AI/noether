#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
from torch import nn

from noether.core.schemas.modules.layers import RopeFrequencyConfig
from noether.modeling.models.ab_upt import AnchoredBranchedUPT
from noether.modeling.modules.blocks import PerceiverBlock, TransformerBlock
from noether.modeling.modules.layers import RopeFrequency
from tutorial.model.base import BaseModel
from tutorial.schemas.models import ABUPTConfig

from .schemas import AnchorBranchedUPTConfig


class AnchoredBranchedUPTRoNE(AnchoredBranchedUPT):
    """
    Extended implementation of the Anchored Branched UPT model with Rotary Normal Embedding (RoNE).

    This version adds support for:
    - Surface normal embeddings via RoPE
    - Normal-aware attention in physics blocks
    - Position-only mode for cross-attention to preserve relative encoding
    """

    def __init__(
        self,
        config: AnchorBranchedUPTConfig,
    ):
        """Initialize the extended AB-UPT model with normal embedding support."""
        super().__init__(config=config)

        head_dim = config.transformer_block_config.hidden_dim // config.transformer_block_config.num_heads
        self.use_normal_rope = config.use_normal_rope
        self.cross_attention_normal_mode = config.cross_attention_normal_mode
        # Parallel list tracking physics block types for per-block RoPE routing in physics_blocks_forward
        self.physics_block_types: list[str] = list(config.physics_blocks)
        self.use_surface_normal_features = config.use_surface_normal_features
        self.surface_normal_embed: nn.Linear | None = (
            nn.Linear(config.data_specs.position_dim, config.hidden_dim) if config.use_surface_normal_features else None
        )

        if self.use_normal_rope:
            normal_head_dim = int(head_dim * config.normal_rope_dim_fraction)
            # ensure even dimension for complex pair representation
            normal_head_dim = normal_head_dim - (normal_head_dim % 2)
            # RopeFrequency needs at least 2 dims per input coordinate axis
            min_normal_dim = 2 * config.data_specs.position_dim
            if normal_head_dim < min_normal_dim:
                raise ValueError(
                    f"normal_rope_dim_fraction={config.normal_rope_dim_fraction} with head_dim={head_dim} "
                    f"produces normal_head_dim={normal_head_dim}, but minimum is {min_normal_dim} "
                    f"for position_dim={config.data_specs.position_dim}. Increase fraction or head_dim."
                )
            position_head_dim = head_dim - normal_head_dim
            self.normal_rope: RopeFrequency | None = RopeFrequency(
                config=RopeFrequencyConfig(
                    hidden_dim=normal_head_dim,
                    input_dim=config.data_specs.position_dim,
                    max_wavelength=int(config.normal_rope_max_wavelength),
                    implementation="complex",
                )  # type: ignore[call-arg]
            )
        else:
            position_head_dim = head_dim
            self.normal_rope = None

        # Override parent's rope with position-only version (smaller head_dim when normals are used)
        self.rope = RopeFrequency(
            config=RopeFrequencyConfig(
                hidden_dim=position_head_dim,
                input_dim=config.data_specs.position_dim,
                implementation="complex",
            )  # type: ignore[call-arg]
        )

    def geometry_branch_forward(
        self,
        geometry_position: torch.Tensor,
        geometry_supernode_idx: torch.Tensor,
        geometry_batch_idx: torch.Tensor,
        condition: torch.Tensor | None,
        geometry_attn_kwargs: dict[str, torch.Tensor],
        geometry_normals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the geometry branch of the model.

        Extended to support passing normals as input features to the encoder.
        """

        # encode geometry (only pass normals if encoder expects input features)
        geometry_encoding: torch.Tensor = self.encoder(
            input_pos=geometry_position,
            supernode_idx=geometry_supernode_idx,
            batch_idx=geometry_batch_idx,
            input_features=geometry_normals if self.encoder.input_features_dim is not None else None,
        )
        if len(self.geometry_blocks) > 0:
            # feed supernodes through some transformer blocks
            for block in self.geometry_blocks:
                geometry_encoding = block(
                    geometry_encoding,
                    attn_kwargs=geometry_attn_kwargs,
                    condition=condition,
                )
        return geometry_encoding

    def physics_blocks_forward(
        self,
        surface_position_all: torch.Tensor,
        volume_position_all: torch.Tensor,
        geometry_encoding: torch.Tensor | None,
        physics_token_specs: list,
        physics_attn_kwargs: dict[str, torch.Tensor],
        physics_perceiver_attn_kwargs: dict[str, torch.Tensor],
        condition: torch.Tensor | None,
        surface_normals_all: torch.Tensor | None = None,
        physics_cross_attn_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the physics blocks of the model.

        Although in the AB-UPT paper we only have a perceiver block at the first block, it is possible to have more
        perceiver blocks in the physics blocks that attend to the geometry encoding.

        Extended with:
        - Surface normal embedding support
        - Position-only RoPE for cross-attention blocks when cross_attention_normal_mode="position_only"

        Args:
            surface_position_all: Tensor of shape (B, N_surface_total, D_pos)
            volume_position_all: Tensor of shape (B, N_volume_total, D_pos)
            geometry_encoding: Tensor of shape (B, N_supernodes, D_hidden)
            physics_token_specs: List of TokenSpec defining the token specifications for the physics blocks.
            physics_attn_kwargs: Additional attention kwargs for the physics transformer blocks.
            physics_perceiver_attn_kwargs: Additional attention kwargs for the physics perceiver blocks.
            condition: Optional conditioning tensor of shape (B, D_condition)
            surface_normals_all: Optional surface normals for surface tokens. Tensor of shape
                (B, N_surface_total, 3). When use_surface_normal_features=True, projected to
                hidden_dim and added to the surface position embedding.
            physics_cross_attn_kwargs: Attention kwargs for CrossAnchorAttention blocks. When
                cross_attention_normal_mode="position_only", contains position-only RoPE freqs so
                that the relative-encoding guarantee is preserved across the surface↔volume boundary.
                Falls back to physics_attn_kwargs when None.
        """

        if not (surface_position_all.ndim == 3 and volume_position_all.ndim == 3):
            raise ValueError("surface_position_all and volume_position_all must be 3-dimensional tensors.")

        surface_all_pos_embed = self.surface_bias(self.pos_embed(surface_position_all))
        if self.surface_normal_embed is not None and surface_normals_all is not None:
            surface_all_pos_embed = surface_all_pos_embed + self.surface_normal_embed(surface_normals_all)
        volume_all_pos_embed = self.volume_bias(self.pos_embed(volume_position_all))
        x_physics = torch.concat([surface_all_pos_embed, volume_all_pos_embed], dim=1)

        for i, block in enumerate(self.physics_blocks):
            if isinstance(block, TransformerBlock):
                # Use position-only freqs for cross blocks when requested
                if physics_cross_attn_kwargs is not None and self.physics_block_types[i] == "cross":
                    chosen_kwargs = physics_cross_attn_kwargs
                else:
                    chosen_kwargs = physics_attn_kwargs
                x_physics = block(
                    x_physics,
                    attn_kwargs=dict(token_specs=physics_token_specs, **chosen_kwargs),
                    condition=condition,
                )
            elif isinstance(block, PerceiverBlock):
                x_physics = block(
                    q=x_physics,
                    kv=geometry_encoding,
                    attn_kwargs=physics_perceiver_attn_kwargs,
                    condition=condition,
                )
            else:
                raise NotImplementedError(f"Unknown block type: {type(block)}")

        return x_physics

    def _compute_rope_freqs(
        self,
        positions: torch.Tensor,
        normals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute combined position + normal RoPE frequencies.

        Args:
            positions: Coordinate tensor for position RoPE.
            normals: Optional surface normal vectors for normal RoPE. If None and use_normal_rope
                is True, uses zero normals which produce identity rotation (no normal encoding).

        Returns:
            Complex frequency tensor of shape (*positions.shape[:-1], head_dim // 2).
        """
        pos_freqs: torch.Tensor = self.rope(positions)
        if self.normal_rope is None:
            return pos_freqs
        if normals is not None:
            normal_freqs = self.normal_rope(normals)
        else:
            # Zero normals → e^(i*0) = 1+0j → identity rotation (no normal encoding)
            zero_normals = torch.zeros_like(positions)
            normal_freqs = self.normal_rope(zero_normals)
        return torch.cat([pos_freqs, normal_freqs], dim=-1)

    def create_rope_frequencies(
        self,
        geometry_position: torch.Tensor,
        geometry_supernode_idx: torch.Tensor,
        surface_position_all: torch.Tensor,
        volume_position_all: torch.Tensor,
        geometry_normals: torch.Tensor | None = None,
        surface_normals_all: torch.Tensor | None = None,
    ):
        """Create RoPE frequencies for all relevant positions (and optionally normals).

        Args:
            geometry_position: Tensor of shape (B * N_geometry, D_pos), sparse tensor.
            geometry_supernode_idx: Tensor of shape (B * number of super nodes,) with indices of supernodes
            surface_position_all: Tensor of shape (B, N_surface_total, D_pos)
            volume_position_all: Tensor of shape (B, N_volume_total, D_pos)
            geometry_normals: Optional sparse tensor of shape (B * N_geometry, 3) with surface normals.
            surface_normals_all: Optional tensor of shape (B, N_surface_total, 3) with surface normals.
        """

        # kwargs for the rope attention
        batch_size = surface_position_all.size(0)
        geometry_attn_kwargs = {}
        surface_decoder_attn_kwargs = {}
        volume_decoder_attn_kwargs = {}
        physics_perceiver_attn_kwargs = {}
        physics_attn_kwargs = {}

        # Geometry supernodes: position + normal freqs
        supernode_pos = geometry_position[geometry_supernode_idx].unsqueeze(0)
        supernode_normals = (
            geometry_normals[geometry_supernode_idx].unsqueeze(0) if geometry_normals is not None else None
        )
        geometry_rope = self._compute_rope_freqs(supernode_pos, supernode_normals)
        channels = geometry_rope.shape[-1]
        geometry_rope = geometry_rope.view(batch_size, -1, channels)
        geometry_attn_kwargs["freqs"] = geometry_rope

        # Surface: position + normal freqs
        rope_surface_all = self._compute_rope_freqs(surface_position_all, surface_normals_all)
        # Volume: position only (normals=None → identity padding when use_normal_rope=True)
        rope_volume_all = self._compute_rope_freqs(volume_position_all, normals=None)

        rope_all = torch.concat([rope_surface_all, rope_volume_all], dim=1)
        surface_decoder_attn_kwargs["freqs"] = rope_surface_all
        physics_perceiver_attn_kwargs["q_freqs"] = rope_all
        physics_perceiver_attn_kwargs["k_freqs"] = geometry_rope
        volume_decoder_attn_kwargs["freqs"] = rope_volume_all
        physics_attn_kwargs["freqs"] = rope_all

        # Cross-attention freqs: when position_only mode, suppress normal freqs on both sides so
        # that RoPE's relative-encoding guarantee holds (asymmetric rotation would leak absolute
        # surface-normal information into the cross-attention logit).
        physics_cross_attn_kwargs: dict[str, torch.Tensor] = {}
        if self.use_normal_rope and self.cross_attention_normal_mode == "position_only":
            rope_surface_posonly = self._compute_rope_freqs(surface_position_all, normals=None)
            rope_all_posonly = torch.concat([rope_surface_posonly, rope_volume_all], dim=1)
            physics_cross_attn_kwargs["freqs"] = rope_all_posonly
        else:
            # "zeros" mode: reuse the same rope_all (existing behaviour)
            physics_cross_attn_kwargs["freqs"] = rope_all

        return (
            geometry_attn_kwargs,
            surface_decoder_attn_kwargs,
            volume_decoder_attn_kwargs,
            physics_perceiver_attn_kwargs,
            physics_attn_kwargs,
            physics_cross_attn_kwargs,
        )

    def forward(
        self,
        # geometry
        geometry_position: torch.Tensor,
        geometry_supernode_idx: torch.Tensor,
        geometry_batch_idx: torch.Tensor | None,
        # anchors
        surface_anchor_position: torch.Tensor,
        volume_anchor_position: torch.Tensor,
        # design parameters
        geometry_design_parameters: torch.Tensor | None = None,
        inflow_design_parameters: torch.Tensor | None = None,
        # queries
        query_surface_position: torch.Tensor | None = None,
        query_volume_position: torch.Tensor | None = None,
        # normals (for RoNE and SupernodePooling input_features)
        geometry_normals: torch.Tensor | None = None,
        surface_anchor_normals: torch.Tensor | None = None,
        query_surface_normals: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the AB-UPT model with Rotary Normal Embedding support.

        Args:
            geometry_position: Coordinates of the geometry mesh. Tensor of shape (B * N_geometry, D_pos), sparse tensor
            geometry_supernode_idx: Indices of the supernodes for the geometry points. Tensor of shape (B * number of super nodes,)
            geometry_batch_idx: Batch indices for the geometry points. Tensor of shape (B * N_geometry,). If None, assumes all points belong to the same batch.
            surface_anchor_position: Coordinates of the surface anchor points. Tensor of shape (B, N_surface_anchor, D_pos)
            volume_anchor_position: Coordinates of the volume anchor points. Tensor of shape (B, N_volume_anchor, D_pos)
            geometry_design_parameters: Design parameters related to the geometry to condition on. Tensor of shape (B, D_geom)
            inflow_design_parameters: Design parameters related to the inflow to condition on. Tensor of shape (B, D_inflow).
            query_surface_position: Coordinates of the query surface points.
            query_volume_position: Coordinates of the query volume points.
            geometry_normals: Surface normals for geometry mesh. Tensor of shape (B * N_geometry, 3), sparse tensor. Used for RoNE and SupernodePooling input_features.
            surface_anchor_normals: Surface normals at anchor points. Tensor of shape (B, N_surface_anchor, 3).
            query_surface_normals: Surface normals at query points. Tensor of shape (B, N_surface_queries, 3).
        """
        condition = self._prepare_condition(geometry_design_parameters, inflow_design_parameters)

        # Create token specifications
        physics_token_specs, surface_token_specs, volume_token_specs = self._create_physics_token_specs(
            surface_position=surface_anchor_position,
            volume_position=volume_anchor_position,
            query_surface_position=query_surface_position,
            query_volume_position=query_volume_position,
        )

        # Concatenate positions for surface and volume
        if query_surface_position is None:
            surface_position_all = surface_anchor_position
        else:
            surface_position_all = torch.concat([surface_anchor_position, query_surface_position], dim=1)

        if query_volume_position is None:
            volume_position_all = volume_anchor_position
        else:
            volume_position_all = torch.concat([volume_anchor_position, query_volume_position], dim=1)

        # Concatenate surface normals (same pattern as positions)
        if surface_anchor_normals is not None:
            if query_surface_normals is None:
                surface_normals_all = surface_anchor_normals
            else:
                surface_normals_all = torch.concat([surface_anchor_normals, query_surface_normals], dim=1)
        else:
            surface_normals_all = None

        # rope frequencies
        (
            geometry_attn_kwargs,
            surface_decoder_attn_kwargs,
            volume_decoder_attn_kwargs,
            physics_perceiver_attn_kwargs,
            physics_attn_kwargs,
            physics_cross_attn_kwargs,
        ) = self.create_rope_frequencies(
            geometry_position,
            geometry_supernode_idx,
            surface_position_all,
            volume_position_all,
            geometry_normals=geometry_normals,
            surface_normals_all=surface_normals_all,
        )
        # geometry branch
        geometry_encoding = None
        if self.use_geometry_branch:
            assert geometry_batch_idx is not None, "geometry_batch_idx must be provided when using the geometry branch."
            geometry_encoding = self.geometry_branch_forward(
                geometry_position=geometry_position,
                geometry_supernode_idx=geometry_supernode_idx,
                geometry_batch_idx=geometry_batch_idx,
                condition=condition,
                geometry_attn_kwargs=geometry_attn_kwargs,
                geometry_normals=geometry_normals,
            )

        # physics blocks
        x_physics = self.physics_blocks_forward(
            surface_position_all=surface_position_all,
            volume_position_all=volume_position_all,
            geometry_encoding=geometry_encoding,
            physics_token_specs=physics_token_specs,
            physics_attn_kwargs=physics_attn_kwargs,
            physics_perceiver_attn_kwargs=physics_perceiver_attn_kwargs,
            condition=condition,
            surface_normals_all=surface_normals_all,
            physics_cross_attn_kwargs=physics_cross_attn_kwargs,
        )
        # decoder blocks
        surface_predictions, volume_predictions = self.decoder_blocks_forward(
            x_physics=x_physics,
            physics_token_specs=physics_token_specs,
            surface_token_specs=surface_token_specs,
            volume_token_specs=volume_token_specs,
            surface_position_all=surface_position_all,
            volume_position_all=volume_position_all,
            surface_decoder_attn_kwargs=surface_decoder_attn_kwargs,
            volume_decoder_attn_kwargs=volume_decoder_attn_kwargs,
            condition=condition,
        )

        predictions = self._slice_predictions(
            surface_predictions=surface_predictions,
            volume_predictions=volume_predictions,
            surface_position=surface_anchor_position,
            volume_position=volume_anchor_position,
            use_surface_queries=query_surface_position is not None,
            use_volume_queries=query_volume_position is not None,
        )
        return predictions


class ABUPT(BaseModel):
    """Implementation of the AB-UPT model."""

    def __init__(
        self,
        model_config: ABUPTConfig,
        **kwargs,
    ):
        """Initialize the AB-UPT model.

        Args:
            model_config: The configuration for the AB-UPT model.
        """

        super().__init__(model_config=model_config, **kwargs)

        self.ab_upt = AnchoredBranchedUPTRoNE(
            config=model_config,
        )

    def forward(
        # geometry
        self,
        geometry_position: torch.Tensor,
        geometry_supernode_idx: torch.Tensor,
        geometry_batch_idx: torch.Tensor | None,
        # anchors
        surface_anchor_position: torch.Tensor,
        volume_anchor_position: torch.Tensor,
        # design parameters
        geometry_design_parameters: torch.Tensor | None = None,
        inflow_design_parameters: torch.Tensor | None = None,
        # queries
        query_surface_position: torch.Tensor | None = None,
        query_volume_position: torch.Tensor | None = None,
        # normals (for RoNE and SupernodePooling input_features)
        geometry_normals: torch.Tensor | None = None,
        surface_anchor_normals: torch.Tensor | None = None,
        query_surface_normals: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the AB-UPT model.

        Args:
            geometry_position: Positions of the geometry points.
            geometry_supernode_idx: Indices of the supernodes for the geometry points.
            geometry_batch_idx: Batch indices for the geometry points.
            surface_anchor_position: Positions of the surface anchor points.
            volume_anchor_position: Positions of the volume anchor points.
            geometry_design_parameters: Design parameters for the geometry.
            inflow_design_parameters: Design parameters for the inflow.
            query_surface_position: Query positions for the surface points.
            query_volume_position: Query positions for the volume points.
            geometry_normals: Surface normals for geometry mesh points.
            surface_anchor_normals: Surface normals at anchor points.
            query_surface_normals: Surface normals at query points.

        Returns:
            A dictionary containing the model outputs.
        """

        return self.ab_upt(
            # geometry
            geometry_position=geometry_position,
            geometry_supernode_idx=geometry_supernode_idx,
            geometry_batch_idx=geometry_batch_idx,
            # anchors
            surface_anchor_position=surface_anchor_position,
            volume_anchor_position=volume_anchor_position,
            # design parameters
            geometry_design_parameters=geometry_design_parameters,
            inflow_design_parameters=inflow_design_parameters,
            # queries
            query_surface_position=query_surface_position,
            query_volume_position=query_volume_position,
            # normals
            geometry_normals=geometry_normals,
            surface_anchor_normals=surface_anchor_normals,
            query_surface_normals=query_surface_normals,
        )
