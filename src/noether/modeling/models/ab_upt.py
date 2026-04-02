#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import copy
from typing import Any

import torch
from torch import Tensor, nn

KVPair = dict[str, Tensor]  # {"k": tensor, "v": tensor}
LayerCache = dict[str, KVPair]  # {token_name: KVPair}
ModelKVCache = dict[str, list[LayerCache]]  # {branch_name: [LayerCache, ...]}

from noether.core.schemas.models import AnchorBranchedUPTConfig
from noether.core.schemas.modules.attention import TokenSpec
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
    Implementation of the Anchored Branched UPT model. Including input embedding and output projection, so this is an off-the-shelf model that can be used directly by providing the appropriate input tensors.
    """

    def __init__(
        self,
        config: AnchorBranchedUPTConfig,
    ):
        """

        Args:
            config: Configuration for the AB-UPT model. See :class:`~noether.core.schemas.models.AnchorBranchedUPTConfig` for details."""
        super().__init__()

        # move this to schema?
        self.data_specs = config.data_specs
        if config.data_specs.conditioning_dims is not None and config.data_specs.conditioning_dims.total_dim > 0:
            condition_dim = config.data_specs.conditioning_dims.total_dim
        else:
            condition_dim = None

        config.transformer_block_config.condition_dim = condition_dim

        if not config.transformer_block_config.use_rope:
            raise ValueError("AB-UPT requires RoPE to be enabled in the transformer block config.")

        self.rope = RopeFrequency(config=config.rope_frequency_config)  # type: ignore[arg-type]
        self.pos_embed = ContinuousSincosEmbed(config=config.pos_embed_config)  # type: ignore[arg-type]

        # geometry
        self.encoder = SupernodePooling(config=config.supernode_pooling_config)  # type: ignore[arg-type]

        self.geometry_blocks = nn.ModuleList(
            [TransformerBlock(config=config.transformer_block_config) for _ in range(config.geometry_depth)],
        )

        # position bias
        self.surface_bias = MLP(config=config.bias_mlp_config)  # type: ignore[arg-type]
        self.volume_bias = MLP(config=config.bias_mlp_config)  # type: ignore[arg-type]

        # physics blocks
        self.num_perceivers = 0
        self.physics_blocks = nn.ModuleList()
        self.use_geometry_branch = False
        for block in config.physics_blocks:
            if block == "perceiver":
                self.use_geometry_branch = True
                perceiver_block = PerceiverBlock(config=config.perceiver_block_config)  # type: ignore[arg-type]
                self.physics_blocks.append(perceiver_block)  # type: ignore[arg-type]
            else:
                if block == "self":
                    attention_constructor = SelfAnchorAttention  # type: ignore[assignment]
                elif block == "cross":
                    attention_constructor = CrossAnchorAttention  # type: ignore[assignment]
                elif block == "joint":
                    attention_constructor = JointAnchorAttention  # type: ignore[assignment]
                else:
                    raise NotImplementedError(
                        f"Unknown physics block type: {block}. Supported: self, cross, joint, perceiver."
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

        # output projection
        self.surface_decoder = LinearProjection(config=config.surface_decoder_config)  # type: ignore[arg-type]
        self.volume_decoder = LinearProjection(config=config.volume_decoder_config)  # type: ignore[arg-type]

    def _slice_predictions(
        self,
        surface_predictions: Tensor | None,
        volume_predictions: Tensor | None,
        num_surface_anchors: int,
        num_volume_anchors: int,
        use_cached_kv: bool = False,
    ) -> dict[str, Tensor]:
        """Slice the predictions from the surface and volume decoders into the appropriate fields according to the data specifications. If queries are used, slice the predictions for anchors and queries separately."""

        def split_anchor_query(preds: Tensor | None, num_anchors: int, domain: str) -> dict[str, Tensor]:
            if preds is None:
                return {}
            if use_cached_kv:  # when using cached KV, all predictions are queries, anchors are not recomputed
                return {f"query_{domain}": preds}
            if preds.size(1) == num_anchors:  # all predictions are anchors, no queries
                return {f"{domain}": preds}
            return {f"{domain}": preds[:, :num_anchors], f"query_{domain}": preds[:, num_anchors:]}

        # split into anchors and queries
        surface_chunks = split_anchor_query(surface_predictions, num_surface_anchors, "surface")
        volume_chunks = split_anchor_query(volume_predictions, num_volume_anchors, "volume")

        predictions: dict[str, Tensor] = {}
        # surface predictions
        for prefix, tensor in surface_chunks.items():
            for name, slc in self.data_specs.surface_output_dims.field_slices.items():
                predictions[f"{prefix}_{name}"] = tensor[..., slc]

        # volume predictions
        for prefix, tensor in volume_chunks.items():
            for name, slc in self.data_specs.volume_output_dims.field_slices.items():  # type: ignore[union-attr]
                predictions[f"{prefix}_{name}"] = tensor[..., slc]
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
        surface_position: torch.Tensor | None,
        volume_position: torch.Tensor | None,
        query_surface_position: torch.Tensor | None = None,
        query_volume_position: torch.Tensor | None = None,
        use_cached_kv: bool = False,
    ) -> tuple[list[TokenSpec], list[TokenSpec], list[TokenSpec]]:
        """Create token specifications for the physics model from input tensors."""
        surface_anchor_size = None if use_cached_kv else surface_position.size(1)  # type: ignore[union-attr]
        volume_anchor_size = None if use_cached_kv else volume_position.size(1)  # type: ignore[union-attr]

        surface_token_specs = [TokenSpec(name="surface_anchors", size=surface_anchor_size)]
        if query_surface_position is not None:
            surface_token_specs.append(TokenSpec(name="surface_queries", size=query_surface_position.size(1)))
        volume_token_specs = [TokenSpec(name="volume_anchors", size=volume_anchor_size)]
        if query_volume_position is not None:
            volume_token_specs.append(TokenSpec(name="volume_queries", size=query_volume_position.size(1)))

        token_specs: list[TokenSpec] = []
        token_specs.extend(surface_token_specs)
        token_specs.extend(volume_token_specs)

        return token_specs, surface_token_specs, volume_token_specs

    def _split_surface_volume_tensors(
        self, tensor: torch.Tensor, token_specs: list[TokenSpec]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split tensor into surface and volume parts. Cached tokens (size=None) are not in the tensor and are skipped."""
        input_specs = [spec for spec in token_specs if spec.size is not None]
        splits = tensor.split([spec.size for spec in input_specs], dim=1)
        token_dict = {spec.name: split for spec, split in zip(input_specs, splits, strict=True)}

        surface_tensors = [token_dict[spec.name] for spec in input_specs if spec.name.startswith("surface")]
        volume_tensors = [token_dict[spec.name] for spec in input_specs if spec.name.startswith("volume")]

        # Handle empty tensors (when using cache, might have no surface or volume queries)
        if not surface_tensors:
            surface_tensors = [torch.empty(tensor.size(0), 0, tensor.size(2), device=tensor.device)]
        if not volume_tensors:
            volume_tensors = [torch.empty(tensor.size(0), 0, tensor.size(2), device=tensor.device)]

        return torch.cat(surface_tensors, dim=1), torch.cat(volume_tensors, dim=1)

    def geometry_branch_forward(
        self,
        geometry_position: torch.Tensor,
        geometry_supernode_idx: torch.Tensor,
        geometry_batch_idx: torch.Tensor,
        condition: torch.Tensor | None,
        geometry_attn_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass through the geometry branch of the model."""

        # encode geometry
        geometry_encoding: torch.Tensor = self.encoder(
            input_pos=geometry_position,
            supernode_idx=geometry_supernode_idx,
            batch_idx=geometry_batch_idx,
        )
        if len(self.geometry_blocks) > 0:
            # feed supernodes through some transformer blocks
            for block in self.geometry_blocks:
                geometry_encoding, _ = block(
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
        physics_attn_kwargs: dict[str, Any],
        physics_perceiver_attn_kwargs: dict[str, Any],
        condition: torch.Tensor | None,
        kv_cache: ModelKVCache | None = None,
    ) -> tuple[torch.Tensor, list[LayerCache]]:
        """
        Forward pass through the physics blocks of the model.
        Although in the AB-UPT paper we only have a perceiver block as the first block, it is possible to have more perceiver blocks in the physics blocks that attend to the geometry encoding.
        """
        physics_cache = kv_cache.get("physics", []) if kv_cache else []
        assert len(physics_cache) in (0, len(self.physics_blocks)), (
            f"physics_cache length ({len(physics_cache)}) must match number of physics blocks ({len(self.physics_blocks)})"
        )

        if not (surface_position_all.ndim == 3 and volume_position_all.ndim == 3):
            raise ValueError("surface_position_all and volume_position_all must be 3-dimensional tensors.")

        surface_all_pos_embed = self.surface_bias(self.pos_embed(surface_position_all))
        volume_all_pos_embed = self.volume_bias(self.pos_embed(volume_position_all))
        x_physics = torch.concat([surface_all_pos_embed, volume_all_pos_embed], dim=1)

        new_physics_cache: list[LayerCache] = []
        for i, block in enumerate(self.physics_blocks):
            if isinstance(block, TransformerBlock):
                x_physics, block_cache = block(
                    x_physics,
                    attn_kwargs=dict(
                        token_specs=physics_token_specs,
                        kv_cache=physics_cache[i] if physics_cache else None,
                        **physics_attn_kwargs,
                    ),
                    condition=condition,
                )
                if block_cache is not None:
                    new_physics_cache.append(block_cache)
            elif isinstance(block, PerceiverBlock):
                x_physics, block_cache = block(
                    q=x_physics,
                    kv=geometry_encoding,
                    attn_kwargs=dict(
                        kv_cache=physics_cache[i]["geometry_encoding"] if physics_cache else None,
                        **physics_perceiver_attn_kwargs,
                    ),
                    condition=condition,
                )
                if block_cache is not None:
                    new_physics_cache.append({"geometry_encoding": block_cache})
            else:
                raise NotImplementedError(f"Unknown block type: {type(block)}")

        return x_physics, new_physics_cache

    def decoder_blocks_forward(
        self,
        x_physics: torch.Tensor,
        physics_token_specs: list[TokenSpec],
        surface_token_specs: list[TokenSpec],
        volume_token_specs: list[TokenSpec],
        surface_decoder_attn_kwargs: dict[str, Any],
        volume_decoder_attn_kwargs: dict[str, Any],
        condition: torch.Tensor | None,
        kv_cache: ModelKVCache | None = None,
        surface_position_all: torch.Tensor | None = None,
        volume_position_all: torch.Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, list[LayerCache], list[LayerCache]]:
        """
        Forward pass through the decoder blocks of the model.

        Returns:
            Tuple of (surface_predictions, volume_predictions, new_surface_cache, new_volume_cache).
        """
        surface_cache = kv_cache.get("surface", []) if kv_cache else []
        volume_cache = kv_cache.get("volume", []) if kv_cache else []
        assert len(surface_cache) in (0, len(self.surface_decoder_blocks)), (
            f"surface_cache length ({len(surface_cache)}) must match number of surface blocks ({len(self.surface_decoder_blocks)})"
        )
        assert len(volume_cache) in (0, len(self.volume_decoder_blocks)), (
            f"volume_cache length ({len(volume_cache)}) must match number of volume blocks ({len(self.volume_decoder_blocks)})"
        )

        x_surface, x_volume = self._split_surface_volume_tensors(x_physics, physics_token_specs)

        # Validate sizes
        if surface_position_all is not None:
            assert x_surface.size(1) == surface_position_all.size(1), (
                "Surface tensor size does not match surface position size."
            )
        if volume_position_all is not None:
            assert x_volume.size(1) == volume_position_all.size(1), (
                "Volume tensor size does not match volume position size."
            )

        new_surface_cache: list[LayerCache] = []
        new_volume_cache: list[LayerCache] = []

        # Surface decoder blocks — only process if we have surface tokens
        surface_predictions: Tensor | None = None
        if x_surface.size(1) > 0:
            for i, block in enumerate(self.surface_decoder_blocks):
                x_surface, block_cache = block(
                    x_surface,
                    attn_kwargs=dict(
                        token_specs=surface_token_specs,
                        kv_cache=surface_cache[i] if surface_cache else None,
                        **surface_decoder_attn_kwargs,
                    ),
                    condition=condition,
                )
                if block_cache is not None:
                    new_surface_cache.append(block_cache)
            surface_predictions = self.surface_decoder(x_surface)

        # Volume decoder blocks — only process if we have volume tokens
        volume_predictions: Tensor | None = None
        if x_volume.size(1) > 0:
            for i, block in enumerate(self.volume_decoder_blocks):
                x_volume, block_cache = block(
                    x_volume,
                    attn_kwargs=dict(
                        token_specs=volume_token_specs,
                        kv_cache=volume_cache[i] if volume_cache else None,
                        **volume_decoder_attn_kwargs,
                    ),
                    condition=condition,
                )
                if block_cache is not None:
                    new_volume_cache.append(block_cache)
            volume_predictions = self.volume_decoder(x_volume)

        return surface_predictions, volume_predictions, new_surface_cache, new_volume_cache

    def create_rope_frequencies(
        self,
        surface_position_all: torch.Tensor,
        volume_position_all: torch.Tensor,
        geometry_position: torch.Tensor | None = None,
        geometry_supernode_idx: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Create RoPE frequencies for all relevant positions."""
        batch_size = surface_position_all.size(0)
        geometry_attn_kwargs: dict[str, Any] = {}
        surface_decoder_attn_kwargs: dict[str, Any] = {}
        volume_decoder_attn_kwargs: dict[str, Any] = {}
        physics_perceiver_attn_kwargs: dict[str, Any] = {}
        physics_attn_kwargs: dict[str, Any] = {}

        if geometry_position is not None and geometry_supernode_idx is not None:
            geometry_rope = self.rope(geometry_position[geometry_supernode_idx].unsqueeze(0))
            channels = geometry_rope.shape[-1]
            geometry_rope = geometry_rope.view(batch_size, -1, channels)
            geometry_attn_kwargs["freqs"] = geometry_rope
            physics_perceiver_attn_kwargs["k_freqs"] = geometry_rope

        rope_surface_all = self.rope(surface_position_all)
        rope_volume_all = self.rope(volume_position_all)
        rope_all = torch.concat([rope_surface_all, rope_volume_all], dim=1)

        surface_decoder_attn_kwargs["freqs"] = rope_surface_all
        physics_perceiver_attn_kwargs["q_freqs"] = rope_all
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
        geometry_position: torch.Tensor | None = None,
        geometry_supernode_idx: torch.Tensor | None = None,
        geometry_batch_idx: torch.Tensor | None = None,
        # anchors
        surface_anchor_position: torch.Tensor | None = None,
        volume_anchor_position: torch.Tensor | None = None,
        # design parameters
        geometry_design_parameters: torch.Tensor | None = None,
        inflow_design_parameters: torch.Tensor | None = None,
        # queries
        query_surface_position: torch.Tensor | None = None,
        query_volume_position: torch.Tensor | None = None,
        # KV cache
        kv_cache: ModelKVCache | None = None,
    ) -> tuple[dict[str, Tensor], ModelKVCache]:
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
            kv_cache: KV cache from a previous forward call. When provided, anchor K/V are loaded
                from the cache and geometry/anchor inputs are not required.

        Returns:
            Tuple of (predictions, kv_cache). Predictions is a dictionary containing the predictions for surface and volume fields, sliced according to the data specifications.
        """
        if (surface_anchor_position is None) == (kv_cache is None):
            raise ValueError(
                "Either surface_anchor_position must be provided (no KV cache) or kv_cache must be provided "
                "(with KV cache), but not both."
            )

        use_cached_kv = kv_cache is not None

        # Validate arguments for each mode
        if use_cached_kv:
            assert surface_anchor_position is None, "surface_anchor_position must be None when using KV cache"
            assert volume_anchor_position is None, "volume_anchor_position must be None when using KV cache"
            assert geometry_position is None, "geometry_position must be None when using KV cache"
            assert geometry_supernode_idx is None, "geometry_supernode_idx must be None when using KV cache"
            assert geometry_batch_idx is None, "geometry_batch_idx must be None when using KV cache"
            assert query_surface_position is not None or query_volume_position is not None, (
                "At least one of query_surface_position or query_volume_position must be provided when using KV cache"
            )
        else:
            assert surface_anchor_position is not None, "surface_anchor_position is required without KV cache"
            assert volume_anchor_position is not None, "volume_anchor_position is required without KV cache"
            if self.use_geometry_branch:
                assert geometry_position is not None, "geometry_position is required when using geometry branch"
                assert geometry_supernode_idx is not None, (
                    "geometry_supernode_idx is required when using geometry branch"
                )
                assert geometry_batch_idx is not None, "geometry_batch_idx is required when using geometry branch"

        condition = self._prepare_condition(geometry_design_parameters, inflow_design_parameters)

        # Create token specifications
        physics_token_specs, surface_token_specs, volume_token_specs = self._create_physics_token_specs(
            surface_position=surface_anchor_position,
            volume_position=volume_anchor_position,
            query_surface_position=query_surface_position,
            query_volume_position=query_volume_position,
            use_cached_kv=use_cached_kv,
        )

        # Concatenate anchor + query positions (or just queries when using cached KV)
        if surface_anchor_position is None or query_surface_position is None:
            surface_position_all = (
                surface_anchor_position if surface_anchor_position is not None else query_surface_position
            )
        else:
            surface_position_all = torch.concat([surface_anchor_position, query_surface_position], dim=1)

        if volume_anchor_position is None or query_volume_position is None:
            volume_position_all = (
                volume_anchor_position if volume_anchor_position is not None else query_volume_position
            )
        else:
            volume_position_all = torch.concat([volume_anchor_position, query_volume_position], dim=1)

        # Ensure both are tensors (empty placeholder if a branch has no tokens)
        if surface_position_all is None:
            assert volume_position_all is not None
            surface_position_all = volume_position_all[:, :0]
        if volume_position_all is None:
            assert surface_position_all is not None
            volume_position_all = surface_position_all[:, :0]

        # RoPE frequencies
        (
            geometry_attn_kwargs,
            surface_decoder_attn_kwargs,
            volume_decoder_attn_kwargs,
            physics_perceiver_attn_kwargs,
            physics_attn_kwargs,
        ) = self.create_rope_frequencies(
            surface_position_all=surface_position_all,
            volume_position_all=volume_position_all,
            geometry_position=geometry_position,
            geometry_supernode_idx=geometry_supernode_idx,
        )

        # Geometry branch (skipped in cached mode)
        geometry_encoding = None
        if not use_cached_kv and self.use_geometry_branch:
            # has been validated earlier but need to exclude None type option for type checker
            assert geometry_position is not None
            assert geometry_supernode_idx is not None
            assert geometry_batch_idx is not None
            geometry_encoding = self.geometry_branch_forward(
                geometry_position=geometry_position,
                geometry_supernode_idx=geometry_supernode_idx,
                geometry_batch_idx=geometry_batch_idx,
                condition=condition,
                geometry_attn_kwargs=geometry_attn_kwargs,
            )

        # Physics blocks
        x_physics, new_physics_cache = self.physics_blocks_forward(
            surface_position_all=surface_position_all,
            volume_position_all=volume_position_all,
            geometry_encoding=geometry_encoding,
            physics_token_specs=physics_token_specs,
            physics_attn_kwargs=physics_attn_kwargs,
            physics_perceiver_attn_kwargs=physics_perceiver_attn_kwargs,
            condition=condition,
            kv_cache=kv_cache,
        )

        # Decoder blocks
        surface_predictions, volume_predictions, new_surface_cache, new_volume_cache = self.decoder_blocks_forward(
            x_physics=x_physics,
            physics_token_specs=physics_token_specs,
            surface_token_specs=surface_token_specs,
            volume_token_specs=volume_token_specs,
            surface_decoder_attn_kwargs=surface_decoder_attn_kwargs,
            volume_decoder_attn_kwargs=volume_decoder_attn_kwargs,
            condition=condition,
            kv_cache=kv_cache,
            surface_position_all=surface_position_all,
            volume_position_all=volume_position_all,
        )

        predictions = self._slice_predictions(
            surface_predictions=surface_predictions,
            volume_predictions=volume_predictions,
            num_surface_anchors=surface_anchor_position.size(1) if surface_anchor_position is not None else 0,
            num_volume_anchors=volume_anchor_position.size(1) if volume_anchor_position is not None else 0,
            use_cached_kv=use_cached_kv,
        )

        # Return KV cache: pass through the provided cache, or assemble a new one
        if kv_cache is None:
            new_kv_cache: ModelKVCache = {}
            if new_physics_cache:
                new_kv_cache["physics"] = new_physics_cache
            if new_surface_cache:
                new_kv_cache["surface"] = new_surface_cache
            if new_volume_cache:
                new_kv_cache["volume"] = new_volume_cache
        else:
            new_kv_cache = kv_cache

        return predictions, new_kv_cache
