#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import copy
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn

from noether.core.schemas.modules.untied import UntiedPerceiverBlockConfig, UntiedTransformerBlockConfig
from noether.modeling.modules.untied import UntiedPerceiverBlock, UntiedTransformerBlock

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
    """Implementation of the Anchored Branched UPT (AB-UPT) model.

    This is an off-the-shelf model — it includes input embedding and output projection,
    so it can be used directly by providing the appropriate input tensors. See
    :meth:`forward` for the expected inputs.

    The architecture is fully driven by
    :class:`~noether.core.schemas.models.AnchorBranchedUPTConfig`: the geometry encoder
    depth, the ordering and type of physics blocks, and the per-domain decoder depths
    are all configured there. For a walkthrough of how to assemble a config (and
    concrete YAML examples from the ``aero_cfd`` and ``heat_transfer`` recipes), see
    :doc:`/guides/training/configuring_ab_upt`.
    """

    def __init__(
        self,
        config: AnchorBranchedUPTConfig,
    ):
        """

        Args:
            config: Configuration for the AB-UPT model. See :class:`~noether.core.schemas.models.AnchorBranchedUPTConfig` for details."""
        super().__init__()

        self.data_specs = config.data_specs

        if not config.transformer_block_config.use_rope:
            raise ValueError("AB-UPT requires RoPE to be enabled in the transformer block config.")

        self.rope = RopeFrequency(config=config.rope_frequency_config)  # type: ignore[arg-type]
        self.pos_embed = ContinuousSincosEmbed(config=config.pos_embed_config)  # type: ignore[arg-type]

        # domains (e.g. surface, volume)
        self.domain_names: list[str] = list(config.data_specs.domains.keys())
        self.domain_biases = nn.ModuleDict(
            {name: MLP(config=config.bias_mlp_config) for name in self.domain_names}  # type: ignore[arg-type]
        )

        self.hidden_dim = config.hidden_dim

        # physics blocks (shared weights, parameterized with N branches)
        self.physics_blocks = nn.ModuleList()
        self.use_geometry_branch = False
        attention_constructors = {
            "self": SelfAnchorAttention,
            "cross": CrossAnchorAttention,
            "joint": JointAnchorAttention,
        }
        num_domains = len(self.domain_names)
        for block in config.physics_blocks:
            untied = block.endswith("_untied")
            block_type = block.removesuffix("_untied") if untied else block

            if block_type == "perceiver":
                self.use_geometry_branch = True
                if not untied:
                    self.physics_blocks.append(PerceiverBlock(config=config.perceiver_block_config))  # type: ignore[arg-type]
                else:
                    self.physics_blocks.append(
                        UntiedPerceiverBlock(
                            config=UntiedPerceiverBlockConfig(
                                num_types=num_domains,
                                perceiver_block_config=config.perceiver_block_config,
                            )
                        )
                    )
            elif block_type in attention_constructors:
                block_config = copy.deepcopy(config.transformer_block_config)
                block_config.attention_constructor = attention_constructors[block_type]  # type: ignore[assignment]
                block_config.attention_arguments = {"branches": tuple(self.domain_names)}
                if not untied:
                    self.physics_blocks.append(TransformerBlock(config=block_config))  # type: ignore[arg-type]
                else:
                    self.physics_blocks.append(
                        UntiedTransformerBlock(
                            config=UntiedTransformerBlockConfig(
                                num_types=num_domains,
                                transformer_block=block_config,
                            )
                        )
                    )
            else:
                raise NotImplementedError(
                    f"Unknown physics block type: {block}. "
                    "Supported: self, cross, joint, perceiver (each optionally with _untied suffix)."
                )

        if self.use_geometry_branch and config.supernode_pooling_config is not None:
            # geometry
            self.encoder = SupernodePooling(config=config.supernode_pooling_config)  # type: ignore[arg-type]
            self.geometry_blocks = nn.ModuleList(
                [TransformerBlock(config=config.transformer_block_config) for _ in range(config.geometry_depth)],
            )

        # per-domain decoder blocks (separate weights per domain)
        self.domain_decoder_blocks = nn.ModuleDict()
        for name, num_blocks in config.num_domain_decoder_blocks.items():
            decoder_block_config = copy.deepcopy(config.transformer_block_config)
            decoder_block_config.attention_constructor = SelfAnchorAttention  # type: ignore[assignment]
            decoder_block_config.attention_arguments = {"branches": (name,)}
            self.domain_decoder_blocks[name] = nn.ModuleList(
                [TransformerBlock(config=decoder_block_config) for _ in range(num_blocks)],
            )

        # per-domain output projection
        self.domain_decoder_projections = nn.ModuleDict(
            {
                name: LinearProjection(config=decoder_config)  # type: ignore[arg-type]
                for name, decoder_config in config.domain_decoder_configs.items()  # type: ignore[attr-defined]
            }
        )

    def _slice_predictions(
        self,
        preds: Tensor,
        domain_name: str,
        num_anchors: int,
        use_cached_kv: bool = False,
    ) -> dict[str, Tensor]:
        """Slice a single domain's raw decoder output into named field predictions.

        Splits into anchor and query predictions when both are present.
        Output keys follow the pattern ``{domain}_{field}`` and ``query_{domain}_{field}``.
        """
        field_slices = self.data_specs.domains[domain_name].output_dims.field_slices
        results: dict[str, Tensor] = {}

        if use_cached_kv:
            for field, slc in field_slices.items():
                results[f"query_{domain_name}_{field}"] = preds[..., slc]
        elif preds.size(1) == num_anchors:
            for field, slc in field_slices.items():
                results[f"{domain_name}_{field}"] = preds[..., slc]
        else:
            for field, slc in field_slices.items():
                results[f"{domain_name}_{field}"] = preds[:, :num_anchors, slc]
                results[f"query_{domain_name}_{field}"] = preds[:, num_anchors:, slc]
        return results

    def _prepare_condition(
        self,
        conditioning_inputs: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor | None:
        """Prepare the condition tensor by concatenating all conditioning inputs."""
        if not conditioning_inputs:
            return None

        parts = [v.squeeze(1) if v.ndim == 3 and v.shape[1] == 1 else v for v in conditioning_inputs.values()]
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def _create_token_specs(
        self,
        domain_name: str,
        anchor_position: torch.Tensor | None,
        query_position: torch.Tensor | None = None,
        use_cached_kv: bool = False,
    ) -> list[TokenSpec]:
        """Create token specifications for a single domain."""
        anchor_size = None if use_cached_kv else (anchor_position.size(1) if anchor_position is not None else 0)
        specs = [TokenSpec(name=f"{domain_name}_anchors", size=anchor_size)]
        if query_position is not None:
            specs.append(TokenSpec(name=f"{domain_name}_queries", size=query_position.size(1)))
        return specs

    def _create_all_token_specs(
        self,
        domain_anchor_positions: dict[str, torch.Tensor],
        domain_query_positions: Mapping[str, torch.Tensor | None],
        use_cached_kv: bool = False,
    ) -> tuple[list[TokenSpec], dict[str, list[TokenSpec]]]:
        """Create token specifications for all domains.

        Returns:
            Tuple of (all_token_specs, per_domain_token_specs).
        """
        per_domain_specs = {
            name: self._create_token_specs(
                name,
                anchor_position=domain_anchor_positions.get(name),
                query_position=domain_query_positions.get(name),
                use_cached_kv=use_cached_kv,
            )
            for name in self.domain_names
        }
        all_specs = [spec for specs in per_domain_specs.values() for spec in specs]
        return all_specs, per_domain_specs

    def _split_domain_tensors(self, tensor: torch.Tensor, token_specs: list[TokenSpec]) -> dict[str, torch.Tensor]:
        """Split a concatenated tensor back into per-domain tensors. Cached tokens (size=None) are skipped."""
        input_specs = [spec for spec in token_specs if spec.size is not None]
        splits = tensor.split([spec.size for spec in input_specs], dim=1)
        token_dict = {spec.name: split for spec, split in zip(input_specs, splits, strict=True)}

        result: dict[str, torch.Tensor] = {}
        for name in self.domain_names:
            parts = [token_dict[spec.name] for spec in input_specs if spec.name.startswith(name)]
            if parts:
                result[name] = torch.cat(parts, dim=1)
            else:
                result[name] = torch.empty(tensor.size(0), 0, tensor.size(2), device=tensor.device)
        return result

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
        for block in self.geometry_blocks:
            geometry_encoding, _ = block(
                geometry_encoding,
                attn_kwargs=geometry_attn_kwargs,
                condition=condition,
            )
        return geometry_encoding

    def physics_blocks_forward(
        self,
        domain_positions_all: dict[str, torch.Tensor],
        geometry_encoding: torch.Tensor | None,
        physics_token_specs: list[TokenSpec],
        physics_attn_kwargs: dict[str, Any],
        physics_perceiver_attn_kwargs: dict[str, Any],
        condition: torch.Tensor | None,
        kv_cache: ModelKVCache | None = None,
    ) -> tuple[torch.Tensor, list[LayerCache]]:
        """Forward pass through the physics blocks of the model."""
        physics_cache = kv_cache.get("physics", []) if kv_cache else []
        assert len(physics_cache) in (0, len(self.physics_blocks)), (
            f"physics_cache length ({len(physics_cache)}) must match number of physics blocks ({len(self.physics_blocks)})"
        )

        # Per-domain position embedding + bias, then concatenate in domain order
        # Preallocate tensor for per-domain position embedding + bias, then fill in domain order
        batch_size = next(iter(domain_positions_all.values())).size(0)
        total_tokens = sum(domain_positions_all[name].size(1) for name in self.domain_names)
        x_physics = torch.empty(batch_size, total_tokens, self.hidden_dim, device=next(self.parameters()).device)
        start = 0
        for name in self.domain_names:
            pos = domain_positions_all[name]
            if pos.ndim != 3:
                raise ValueError(f"Position tensor for domain '{name}' must be 3-dimensional, got {pos.ndim}.")
            emb = self.domain_biases[name](self.pos_embed(pos))
            end = start + emb.size(1)
            x_physics[:, start:end, :] = emb
            start = end
        new_physics_cache: list[LayerCache] = []
        for i, block in enumerate(self.physics_blocks):
            if isinstance(block, TransformerBlock | UntiedTransformerBlock):
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
                perceiver_attn_kwargs: dict[str, Any] = dict(
                    kv_cache=physics_cache[i]["geometry_encoding"] if physics_cache else None,
                    **physics_perceiver_attn_kwargs,
                )
                if isinstance(block, UntiedPerceiverBlock):
                    perceiver_attn_kwargs["token_specs"] = physics_token_specs
                x_physics, block_cache = block(
                    q=x_physics,
                    kv=geometry_encoding,
                    attn_kwargs=perceiver_attn_kwargs,
                    condition=condition,
                )
                if block_cache is not None:
                    new_physics_cache.append({"geometry_encoding": block_cache})
            else:
                raise NotImplementedError(f"Unknown block type: {type(block)}")

        return x_physics, new_physics_cache

    def _decode_domain(
        self,
        x: torch.Tensor,
        domain_name: str,
        token_specs: list[TokenSpec],
        decoder_attn_kwargs: dict[str, Any],
        condition: torch.Tensor | None,
        cache: list[LayerCache] | None = None,
    ) -> tuple[Tensor | None, list[LayerCache]]:
        """Run decoder blocks + output projection for a single domain.

        Returns:
            Tuple of (predictions, new_cache). Predictions is None if x has no tokens.
        """
        new_cache: list[LayerCache] = []

        if x.size(1) == 0:
            return None, new_cache

        if domain_name not in self.domain_decoder_blocks:
            decoder_blocks = nn.ModuleList()
        else:
            decoder_blocks = cast("nn.ModuleList", self.domain_decoder_blocks[domain_name])
        if cache is not None:
            assert len(cache) in (0, len(decoder_blocks)), (
                f"{domain_name} cache length ({len(cache)}) must match number of decoder blocks ({len(decoder_blocks)})"
            )

        for i, block in enumerate(decoder_blocks):
            x, block_cache = block(
                x,
                attn_kwargs=dict(token_specs=token_specs, kv_cache=cache[i] if cache else None, **decoder_attn_kwargs),
                condition=condition,
            )
            if block_cache is not None:
                new_cache.append(block_cache)

        return self.domain_decoder_projections[domain_name](x), new_cache

    def decoder_blocks_forward(
        self,
        x_physics: torch.Tensor,
        physics_token_specs: list[TokenSpec],
        per_domain_token_specs: dict[str, list[TokenSpec]],
        decoder_attn_kwargs: dict[str, dict[str, Any]],
        condition: torch.Tensor | None,
        kv_cache: ModelKVCache | None = None,
        domain_positions_all: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, list[LayerCache]]]:
        """Forward pass through the per-domain decoder blocks.

        Returns:
            Tuple of (domain_predictions, new_domain_caches).
        """
        domain_tensors = self._split_domain_tensors(x_physics, physics_token_specs)

        domain_predictions: dict[str, Tensor] = {}
        new_domain_caches: dict[str, list[LayerCache]] = {}

        for name in self.domain_names:
            x_domain = domain_tensors[name]

            # Validate sizes
            if domain_positions_all is not None and name in domain_positions_all:
                assert x_domain.size(1) == domain_positions_all[name].size(1), (
                    f"{name} tensor size does not match {name} position size."
                )

            preds, new_cache = self._decode_domain(
                x_domain,
                name,
                per_domain_token_specs[name],
                decoder_attn_kwargs[name],
                condition,
                cache=kv_cache.get(name, []) if kv_cache else None,
            )
            if preds is not None:
                domain_predictions[name] = preds
            new_domain_caches[name] = new_cache

        return domain_predictions, new_domain_caches

    def create_rope_frequencies(
        self,
        domain_positions_all: dict[str, torch.Tensor],
        geometry_position: torch.Tensor | None = None,
        geometry_supernode_idx: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """Create RoPE frequencies for all relevant positions.

        Returns:
            Tuple of (geometry_attn_kwargs, decoder_attn_kwargs, physics_perceiver_attn_kwargs, physics_attn_kwargs).
            decoder_attn_kwargs is keyed by domain name.
        """
        first_pos = next(iter(domain_positions_all.values()))
        batch_size = first_pos.size(0)

        geometry_attn_kwargs: dict[str, Any] = {}
        physics_perceiver_attn_kwargs: dict[str, Any] = {}
        physics_attn_kwargs: dict[str, Any] = {}

        if geometry_position is not None and geometry_supernode_idx is not None:
            geometry_rope = self.rope(geometry_position[geometry_supernode_idx].unsqueeze(0))
            channels = geometry_rope.shape[-1]
            geometry_rope = geometry_rope.view(batch_size, -1, channels)
            geometry_attn_kwargs["freqs"] = geometry_rope
            physics_perceiver_attn_kwargs["k_freqs"] = geometry_rope

        # Per-domain rope + concatenated physics rope
        domain_rope = {name: self.rope(domain_positions_all[name]) for name in self.domain_names}
        rope_all = torch.cat([domain_rope[name] for name in self.domain_names], dim=1)

        physics_perceiver_attn_kwargs["q_freqs"] = rope_all
        physics_attn_kwargs["freqs"] = rope_all

        decoder_attn_kwargs = {name: {"freqs": domain_rope[name]} for name in self.domain_names}

        return geometry_attn_kwargs, decoder_attn_kwargs, physics_perceiver_attn_kwargs, physics_attn_kwargs

    def forward(
        self,
        # geometry
        geometry_position: torch.Tensor | None = None,
        geometry_supernode_idx: torch.Tensor | None = None,
        geometry_batch_idx: torch.Tensor | None = None,
        # domain positions
        domain_anchor_positions: dict[str, Tensor] | None = None,
        domain_query_positions: dict[str, Tensor] | None = None,
        domain_features: dict[str, Tensor] | None = None,
        conditioning_inputs: dict[str, Tensor] | None = None,
        # KV cache
        kv_cache: ModelKVCache | None = None,
    ) -> tuple[dict[str, Tensor], ModelKVCache]:
        """Forward pass of the AB-UPT model.

        Example::

            model(
                geometry_position=...,
                geometry_supernode_idx=...,
                geometry_batch_idx=...,
                domain_anchor_positions={"surface": surface_pos, "volume": volume_pos},
                domain_query_positions={"surface": query_pos},
                conditioning_inputs={"geometry_design_parameters": design_params},
            )

        Args:
            geometry_position: Coordinates of the geometry mesh. Tensor of shape (B * N_geometry, D_pos).
            geometry_supernode_idx: Supernode indices for the geometry points.
            geometry_batch_idx: Batch indices for the geometry points.
            domain_anchor_positions: Per-domain anchor positions, e.g. ``{"surface": (B, N, D), "volume": (B, M, D)}``.
            domain_query_positions: Per-domain query positions (optional).
            conditioning_inputs: Conditioning tensors, e.g. ``{"geometry_design_parameters": (B, D)}``.
            kv_cache: KV cache from a previous forward call.

        Returns:
            Tuple of (predictions, kv_cache).
        """
        domain_anchor_positions = domain_anchor_positions or {}
        domain_query_positions = domain_query_positions or {}

        use_cached_kv = kv_cache is not None
        has_anchors = bool(domain_anchor_positions)

        # Validate: either anchors or kv_cache, not both, not neither
        if has_anchors == (kv_cache is not None):
            raise ValueError(
                "Either domain anchor positions must be provided (no KV cache) or kv_cache must be provided, but not both."
            )

        if use_cached_kv:
            assert geometry_position is None, "geometry_position must be None when using KV cache"
            assert geometry_supernode_idx is None, "geometry_supernode_idx must be None when using KV cache"
            assert geometry_batch_idx is None, "geometry_batch_idx must be None when using KV cache"
            assert domain_query_positions, "At least one domain query position must be provided when using KV cache"
        else:
            if self.use_geometry_branch:
                assert geometry_position is not None, "geometry_position is required when using geometry branch"
                assert geometry_supernode_idx is not None, (
                    "geometry_supernode_idx is required when using geometry branch"
                )
                assert geometry_batch_idx is not None, "geometry_batch_idx is required when using geometry branch"

        condition = self._prepare_condition(conditioning_inputs)

        # Create token specifications
        physics_token_specs, per_domain_token_specs = self._create_all_token_specs(
            domain_anchor_positions=domain_anchor_positions,
            domain_query_positions=domain_query_positions,
            use_cached_kv=use_cached_kv,
        )

        # Combine anchor + query positions per domain (or just queries when using cached KV)
        domain_positions_all: dict[str, torch.Tensor] = {}
        for name in self.domain_names:
            anchor = domain_anchor_positions.get(name)
            query = domain_query_positions.get(name)
            if anchor is not None and query is not None:
                domain_positions_all[name] = torch.cat([anchor, query], dim=1)
            elif anchor is not None:
                domain_positions_all[name] = anchor
            elif query is not None:
                domain_positions_all[name] = query
            else:
                # Empty placeholder — infer shape from any available tensor
                ref = (
                    next(iter(domain_anchor_positions.values()))
                    if domain_anchor_positions
                    else next(iter(domain_query_positions.values()))
                )
                domain_positions_all[name] = ref[:, :0]

        # RoPE frequencies
        geometry_attn_kwargs, decoder_attn_kwargs, physics_perceiver_attn_kwargs, physics_attn_kwargs = (
            self.create_rope_frequencies(
                domain_positions_all=domain_positions_all,
                geometry_position=geometry_position,
                geometry_supernode_idx=geometry_supernode_idx,
            )
        )

        # Geometry branch (skipped in cached mode)
        geometry_encoding = None
        if not use_cached_kv and self.use_geometry_branch:
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
            domain_positions_all=domain_positions_all,
            geometry_encoding=geometry_encoding,
            physics_token_specs=physics_token_specs,
            physics_attn_kwargs=physics_attn_kwargs,
            physics_perceiver_attn_kwargs=physics_perceiver_attn_kwargs,
            condition=condition,
            kv_cache=kv_cache,
        )

        # Decoder blocks
        domain_predictions, new_domain_caches = self.decoder_blocks_forward(
            x_physics=x_physics,
            physics_token_specs=physics_token_specs,
            per_domain_token_specs=per_domain_token_specs,
            decoder_attn_kwargs=decoder_attn_kwargs,
            condition=condition,
            kv_cache=kv_cache,
            domain_positions_all=domain_positions_all,
        )

        # Slice predictions into named fields
        predictions: dict[str, Tensor] = {}
        for name, preds in domain_predictions.items():
            num_anchors = domain_anchor_positions[name].size(1) if name in domain_anchor_positions else 0
            predictions.update(self._slice_predictions(preds, name, num_anchors, use_cached_kv))

        # Return KV cache
        if kv_cache is None:
            new_kv_cache: ModelKVCache = {}
            if new_physics_cache:
                new_kv_cache["physics"] = new_physics_cache
            for name, cache in new_domain_caches.items():
                if cache:
                    new_kv_cache[name] = cache
        else:
            new_kv_cache = kv_cache

        return predictions, new_kv_cache
