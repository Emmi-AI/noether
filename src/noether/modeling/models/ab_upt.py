#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import copy

import torch
from torch import Tensor, nn

from noether.core.schemas.models import AnchorBranchedUPTConfig
from noether.core.schemas.modules.attention import TokenSpec
from noether.core.schemas.modules.blocks import PerceiverBlockConfig
from noether.core.schemas.modules.layers import (
    ContinuousSincosEmbeddingConfig,
    LinearProjectionConfig,
    RopeFrequencyConfig,
)
from noether.core.schemas.modules.mlp import MLPConfig
from noether.modeling.modules.attention.anchor_attention import (
    CrossAnchorAttention,
    JointAnchorAttention,
    SelfAnchorAttention,
)
from noether.modeling.modules.blocks import PerceiverBlock, TransformerBlock
from noether.modeling.modules.encoders import SupernodePooling
from noether.modeling.modules.layers import ContinuousSincosEmbed, LinearProjection, RopeFrequency
from noether.modeling.modules.mlp import MLP


class AnchoredBranchedUPT(nn.Module):
    """
    Implementation of the Anchored Branched UPT model.
    """

    def __init__(
        self,
        config: AnchorBranchedUPTConfig,
    ):
        """ """
        super().__init__()

        self.data_specs = config.data_specs
        if config.data_specs.conditioning_dims is not None and config.data_specs.conditioning_dims.total_dim > 0:
            condition_dim = config.data_specs.conditioning_dims.total_dim
        else:
            condition_dim = None

        config.transformer_block_config.condition_dim = condition_dim

        if not config.transformer_block_config.use_rope:
            raise ValueError("AB-UPT requires RoPE to be enabled in the transformer block config.")

        self.rope = RopeFrequency(
            config=RopeFrequencyConfig(
                hidden_dim=config.transformer_block_config.hidden_dim // config.transformer_block_config.num_heads,
                input_dim=config.data_specs.position_dim,
                implementation="complex",
            )  # type: ignore[call-arg]
        )  # type: ignore[call-arg]

        # geometry
        self.encoder = SupernodePooling(config=config.supernode_pooling_config)

        self.geometry_blocks = nn.ModuleList(
            [TransformerBlock(config=config.transformer_block_config) for _ in range(config.geometry_depth)],
        )
        # pos_embed
        self.pos_embed = ContinuousSincosEmbed(
            config=ContinuousSincosEmbeddingConfig(
                hidden_dim=config.hidden_dim, input_dim=config.data_specs.position_dim
            )  # type: ignore[call-arg]
        )

        self.surface_bias = MLP(
            config=MLPConfig(
                input_dim=config.hidden_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.hidden_dim,
            )
        )

        self.volume_bias = MLP(
            config=MLPConfig(
                input_dim=config.hidden_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.hidden_dim,
            )
        )

        self.num_perceivers = 0
        self.physics_blocks = nn.ModuleList()
        self.use_geometry_branch = False
        for block in config.physics_blocks:
            if block == "perceiver":
                self.use_geometry_branch = True
                block = PerceiverBlock(
                    config=PerceiverBlockConfig(
                        hidden_dim=config.hidden_dim,
                        num_heads=config.transformer_block_config.num_heads,
                        mlp_expansion_factor=config.transformer_block_config.mlp_expansion_factor,
                        kv_dim=None,
                        use_rope=config.transformer_block_config.use_rope,
                        condition_dim=condition_dim,
                    )  # type: ignore[call-arg]
                )  # type: ignore[assignment]
            else:
                if block == "shared":
                    attention_constructor = SelfAnchorAttention  # type: ignore[assignment]
                elif block == "cross":
                    attention_constructor = CrossAnchorAttention  # type: ignore[assignment]
                elif block == "joint":
                    attention_constructor = JointAnchorAttention  # type: ignore[assignment]
                else:
                    raise NotImplementedError(
                        f"Unknown physics block type: {block}. Supported: shared, cross, joint, perceiver."
                    )

                block_config = copy.deepcopy(config.transformer_block_config)
                block_config.attention_constructor = attention_constructor  # type: ignore[assignment]
                block_config.attention_arguments = {"branches": ("surface", "volume")}
                block = TransformerBlock(config=block_config)  # type: ignore[assignment]
            self.physics_blocks.append(block)  # type: ignore[arg-type]

        # surface decoder blocks
        surface_blocks_config = copy.deepcopy(config.transformer_block_config)  # check if this work
        surface_blocks_config.attention_constructor = SelfAnchorAttention  # type: ignore[assignment]
        surface_blocks_config.attention_arguments = {"branches": ("surface",)}
        self.surface_decoder_blocks = nn.ModuleList(
            [TransformerBlock(config=surface_blocks_config) for _ in range(config.num_surface_blocks)],
        )

        # volume decoder blocks
        volume_blocks_config = copy.deepcopy(config.transformer_block_config)  # check if this work
        volume_blocks_config.attention_constructor = SelfAnchorAttention  # type: ignore[assignment]
        volume_blocks_config.attention_arguments = {"branches": ("volume",)}
        self.volume_decoder_blocks = nn.ModuleList(
            [TransformerBlock(config=volume_blocks_config) for _ in range(config.num_volume_blocks)],
        )

        self.surface_decoder = LinearProjection(
            config=LinearProjectionConfig(
                input_dim=config.hidden_dim,
                output_dim=config.data_specs.surface_output_dims.total_dim,
                init_weights="truncnormal002",
            )  # type: ignore[call-arg]
        )
        self.volume_decoder = LinearProjection(
            config=LinearProjectionConfig(
                input_dim=config.hidden_dim,
                output_dim=config.data_specs.volume_output_dims.total_dim,  # type: ignore[union-attr]
                init_weights="truncnormal002",
            )  # type: ignore[call-arg]
        )

    def _slice_predictions(
        self,
        surface_predictions: Tensor,
        volume_predictions: Tensor,
        surface_position: Tensor,
        volume_position: Tensor,
        use_surface_queries: bool,
        use_volume_queries: bool,
    ) -> dict[str, Tensor]:
        predictions: dict[str, Tensor] = {}
        assert surface_predictions.size(-1) == sum(dict(self.data_specs.surface_output_dims).values())

        surface_field_slices = self.data_specs.surface_output_dims.field_slices  # e.g. {pressure: slice(0, 1), ...}
        if not use_surface_queries:
            for k in dict(self.data_specs.surface_output_dims).keys():
                predictions[f"surface_{k}"] = surface_predictions[..., surface_field_slices[k]]
        else:
            x_anchor_surface = surface_predictions[:, : surface_position.size(1)]
            x_query_surface = surface_predictions[:, surface_position.size(1) :]
            for k in dict(self.data_specs.surface_output_dims).keys():
                predictions[f"surface_{k}"] = x_anchor_surface[..., surface_field_slices[k]]
                predictions[f"query_surface_{k}"] = x_query_surface[..., surface_field_slices[k]]

        assert volume_predictions.size(-1) == self.data_specs.volume_output_dims.total_dim  # type: ignore[union-attr]
        # e.g. {pressure: slice(0, 1), ...}:
        volume_field_slices = self.data_specs.volume_output_dims.field_slices  # type: ignore[union-attr]
        if not use_volume_queries:
            for k in volume_field_slices:
                predictions[f"volume_{k}"] = volume_predictions[..., volume_field_slices[k]]
        else:
            x_anchor_volume = volume_predictions[:, : volume_position.size(1)]
            x_query_volume = volume_predictions[:, volume_position.size(1) :]
            for k in volume_field_slices:
                predictions[f"volume_{k}"] = x_anchor_volume[..., volume_field_slices[k]]
                predictions[f"query_volume_{k}"] = x_query_volume[..., volume_field_slices[k]]
        return predictions

    def _prepare_condition(
        self,
        geometry_design_parameters: torch.Tensor | None,
        inflow_design_parameters: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Prepare the condition tensor by concatenating the appropriate design parameters."""
        # Ensure design parameters have correct dimensions
        if (
            geometry_design_parameters is not None
            and geometry_design_parameters.ndim == 3
            and geometry_design_parameters.shape[1] == 1
        ):
            geometry_design_parameters = geometry_design_parameters.squeeze(1)
        if (
            inflow_design_parameters is not None
            and inflow_design_parameters.ndim == 3
            and inflow_design_parameters.shape[1] == 1
        ):
            inflow_design_parameters = inflow_design_parameters.squeeze(1)

        conditions = []
        if geometry_design_parameters is not None:
            conditions.append(geometry_design_parameters)
        if inflow_design_parameters is not None:
            conditions.append(inflow_design_parameters)

        if not conditions:
            return None

        return torch.cat(conditions, dim=-1) if len(conditions) > 1 else conditions[0]

    def _create_physics_token_specs(
        self,
        surface_position: torch.Tensor,
        volume_position: torch.Tensor,
        query_surface_position: torch.Tensor | None = None,
        query_volume_position: torch.Tensor | None = None,
    ) -> tuple[list[TokenSpec], list[TokenSpec], list[TokenSpec]]:
        """Create token specifications for the physics model from input tensors."""
        token_specs: list[TokenSpec] = []

        surface_token_specs = [TokenSpec(name="surface_anchors", size=surface_position.size(1))]
        if query_surface_position is not None:
            surface_token_specs.append(TokenSpec(name="surface_queries", size=query_surface_position.size(1)))
        volume_token_specs = [TokenSpec(name="volume_anchors", size=volume_position.size(1))]
        if query_volume_position is not None:
            volume_token_specs.append(TokenSpec(name="volume_queries", size=query_volume_position.size(1)))

        token_specs.extend(surface_token_specs)
        token_specs.extend(volume_token_specs)

        return token_specs, surface_token_specs, volume_token_specs

    def _split_tensor_by_token_specs(
        self, tensor: torch.Tensor, token_specs: list[TokenSpec]
    ) -> dict[str, torch.Tensor]:
        """Split tensor according to token specifications."""
        sizes = [spec.size for spec in token_specs]
        splits = tensor.split(sizes, dim=1)
        return {spec.name: split for spec, split in zip(token_specs, splits, strict=True)}

    def _split_surface_volume_tensors(
        self, tensor: torch.Tensor, token_specs: list[TokenSpec]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split tensor into surface and volume parts using slicing (no contiguity assumption)."""
        token_dict = self._split_tensor_by_token_specs(tensor, token_specs)
        surface_tensors = [token_dict[spec.name] for spec in token_specs if spec.name.startswith("surface")]
        volume_tensors = [token_dict[spec.name] for spec in token_specs if spec.name.startswith("volume")]
        return torch.cat(surface_tensors, dim=1), torch.cat(volume_tensors, dim=1)

    def geometry_branch_forward(
        self,
        geometry_position: torch.Tensor,
        geometry_supernode_idx: torch.Tensor,
        geometry_batch_idx: torch.Tensor,
        condition: torch.Tensor | None,
        geometry_attn_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through the geometry branch of the model.
        """

        # encode geometry
        geometry_encoding: torch.Tensor = self.encoder(
            input_pos=geometry_position,
            supernode_idx=geometry_supernode_idx,
            batch_idx=geometry_batch_idx,
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
        physics_token_specs: list[TokenSpec],
        physics_attn_kwargs: dict[str, torch.Tensor],
        physics_perceiver_attn_kwargs: dict[str, torch.Tensor],
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Forward pass through the physics blocks of the model.

        Allthough in the AB-UPT paper we only have a perceiver block a the first block, it is possible to have more perceiver blocks in the physics blocks that attend to the geometry encoding.

        Args:
            surface_position_all: Tensor of shape (B, N_surface_total, D_pos)
            volume_position_all: Tensor of shape (B, N_volume_total, D_pos)
            geometry_encoding: Tensor of shape (B, N_supernodes, D_hidden)
            physics_token_specs: List of TokenSpec defining the token specifications for the physics blocks.
            physics_attn_kwargs: Additional attention kwargs for the physics transformer blocks.
            physics_perceiver_attn_kwargs: Additional attention kwargs for the physics perceiver blocks.
            condition: Optional conditioning tensor of shape (B, D_condition)
        """

        if not (surface_position_all.ndim == 3 and volume_position_all.ndim == 3):
            raise ValueError("surface_position_all and volume_position_all must be 3-dimensional tensors.")

        surface_all_pos_embed = self.surface_bias(self.pos_embed(surface_position_all))
        volume_all_pos_embed = self.volume_bias(self.pos_embed(volume_position_all))
        x_physics = torch.concat([surface_all_pos_embed, volume_all_pos_embed], dim=1)

        for block in self.physics_blocks:
            if isinstance(block, TransformerBlock):
                x_physics = block(
                    x_physics,
                    attn_kwargs=dict(token_specs=physics_token_specs, **physics_attn_kwargs),
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

    def decoder_blocks_forward(
        self,
        x_physics: torch.Tensor,
        physics_token_specs: list[TokenSpec],
        surface_token_specs: list[TokenSpec],
        volume_token_specs: list[TokenSpec],
        surface_position_all: torch.Tensor,
        volume_position_all: torch.Tensor,
        surface_decoder_attn_kwargs: dict[str, torch.Tensor],
        volume_decoder_attn_kwargs: dict[str, torch.Tensor],
        condition: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the decoder blocks of the model. We have a separate decoder for surface and volume tokens.
        """
        # Split physics output into surface and volume tokens

        x_surface, x_volume = self._split_surface_volume_tensors(x_physics, physics_token_specs)
        if not (x_surface.size(1) == surface_position_all.size(1)):
            raise ValueError("Surface tensor size does not match surface position size.")
        if not (x_volume.size(1) == volume_position_all.size(1)):
            raise ValueError("Volume tensor size does not match volume position size.")

        # surface decoder blocks
        for block in self.surface_decoder_blocks:
            x_surface = block(
                x_surface,
                attn_kwargs=dict(token_specs=surface_token_specs, **surface_decoder_attn_kwargs),
                condition=condition,
            )
        surface_predictions = self.surface_decoder(x_surface)

        # volume decoder blocks
        for block in self.volume_decoder_blocks:
            x_volume = block(
                x_volume,
                attn_kwargs=dict(token_specs=volume_token_specs, **volume_decoder_attn_kwargs),
                condition=condition,
            )
        volume_predictions = self.volume_decoder(x_volume)

        return surface_predictions, volume_predictions

    def create_rope_frequencies(
        self,
        geometry_position: torch.Tensor,
        geometry_supernode_idx: torch.Tensor,
        surface_position_all: torch.Tensor,
        volume_position_all: torch.Tensor,
    ):
        """Create RoPE frequencies for all relevant positions.

        Args:
            geometry_position: Tensor of shape (B * N_geometry, D_pos), sparse tensor.
            geometry_supernode_idx: Tensor of shape (B * number of super nodes,) with indices of supernodes
            surface_position_all: Tensor of shape (B, N_surface_total, D_pos)
            volume_position_all: Tensor of shape (B, N_volume_total, D_pos)
        """

        # kwargs for the rope attention
        batch_size = surface_position_all.size(0)
        geometry_attn_kwargs = {}
        surface_decoder_attn_kwargs = {}
        volume_decoder_attn_kwargs = {}
        physics_perceiver_attn_kwargs = {}
        physics_attn_kwargs = {}

        geometry_rope = self.rope(geometry_position[geometry_supernode_idx].unsqueeze(0))
        channels = geometry_rope.shape[-1]
        geometry_rope = geometry_rope.view(batch_size, -1, channels)
        geometry_attn_kwargs["freqs"] = geometry_rope
        rope_surface_all = self.rope(surface_position_all)
        rope_volume_all = self.rope(volume_position_all)
        rope_all = torch.concat([rope_surface_all, rope_volume_all], dim=1)
        surface_decoder_attn_kwargs["freqs"] = rope_surface_all
        physics_perceiver_attn_kwargs["q_freqs"] = rope_all
        physics_perceiver_attn_kwargs["k_freqs"] = geometry_rope
        volume_decoder_attn_kwargs["freqs"] = rope_volume_all
        physics_attn_kwargs["freqs"] = rope_all

        return (
            geometry_attn_kwargs,
            surface_decoder_attn_kwargs,
            volume_decoder_attn_kwargs,
            physics_perceiver_attn_kwargs,
            physics_attn_kwargs,
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
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the AB-UPT model.
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

        # rope frequencies
        (
            geometry_attn_kwargs,
            surface_decoder_attn_kwargs,
            volume_decoder_attn_kwargs,
            physics_perceiver_attn_kwargs,
            physics_attn_kwargs,
        ) = self.create_rope_frequencies(
            geometry_position, geometry_supernode_idx, surface_position_all, volume_position_all
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
