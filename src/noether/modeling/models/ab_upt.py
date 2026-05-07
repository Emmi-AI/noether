#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import copy
from collections.abc import Mapping
from typing import Any, TypedDict, cast

import torch
from torch import Tensor, nn

from noether.core.schemas.modules.layers.linear_projection import LinearProjectionConfig
from noether.core.schemas.modules.untied import UntiedPerceiverBlockConfig, UntiedTransformerBlockConfig
from noether.modeling.functional.modulation import modulate_scale_shift
from noether.modeling.modules.layers.vectors_conditioner import VectorsConditioner
from noether.modeling.modules.untied import UntiedPerceiverBlock, UntiedTransformerBlock

KVPair = dict[str, Tensor]  # {"k": tensor, "v": tensor}
LayerCache = dict[str, KVPair]  # {token_name: KVPair}


class ModelKVCache(TypedDict, total=False):
    """Caches that let a forward call skip work it has already done.

    All keys are independent — pass any subset. ``geometry_encoding`` and
    ``geometry_rope`` travel together (both depend only on the geometry mesh)
    and are what diffusion sampling reuses across Euler steps.

    * ``geometry_encoding`` — output of :meth:`AnchoredBranchedUPT.geometry_branch_forward`.
      When present, the geometry encoder + transformer stack are skipped.
    * ``geometry_rope`` — RoPE of the geometry supernode positions, used by
      perceiver blocks as ``k_freqs``. Cached alongside ``geometry_encoding``
      so subsequent calls don't need ``geometry_position``.
    * ``physics_blocks`` — per-block anchor self-attention K/V (``None`` for
      perceiver blocks, which always re-project from ``geometry_encoding``).
      When present, anchors are not re-encoded; the call is decode-only.
    * ``decoders`` — per-domain anchor K/V for each decoder block. Symmetric
      with ``physics_blocks`` for the per-domain decoder stage.
    """

    geometry_encoding: Tensor
    geometry_rope: Tensor
    physics_blocks: list[LayerCache | None]
    decoders: dict[str, list[LayerCache]]


from noether.core.schemas.models import AnchorBranchedUPTConfig
from noether.core.schemas.modules.attention import TokenSpec
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


class ReadoutLayer(nn.Module):
    """
    The final readout layer for AB-UPT, which applies an AdaLN-style modulation followed by a linear projection to the desired output dimension.
    """

    def __init__(self, decoder_config: LinearProjectionConfig, hidden_dim: int, condition_dim: int | None = None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=condition_dim is None, eps=1e-6)
        self.linear = LinearProjection(config=decoder_config)
        self.modulation = None
        if condition_dim is not None:
            self.modulation = LinearProjection(
                config=LinearProjectionConfig(
                    input_dim=condition_dim, output_dim=hidden_dim * 2, init_weights="zeros", bias=True
                )
            )

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        x = self.norm_final(x)
        if self.modulation is not None and condition is not None:
            shift, scale = self.modulation(condition).chunk(2, dim=1)
            x = modulate_scale_shift(x, shift, scale)
        x = self.linear(x)
        return x


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
                [
                    TransformerBlock(config=config.geometry_transformer_block_config)  # type: ignore[arg-type]
                    for _ in range(config.geometry_depth)
                ],
            )

        # Per-domain physics-feature projections (e.g. noisy fields for diffusion).
        # Driven by ``data_specs.domains[name].feature_dim``: when a domain
        # declares input feature dims, the backbone builds an MLP that projects
        # ``domain_anchor_features[name]`` and/or ``domain_query_features[name]``
        # to ``hidden_dim`` and adds them to the corresponding position
        # embeddings inside ``physics_blocks_forward``. Sides without features
        # are zero-padded.
        self.domain_feature_projs: nn.ModuleDict | None = None
        for name, spec in config.data_specs.domains.items():
            if spec.feature_dim is None or spec.feature_dim.total_dim == 0:
                continue
            if self.domain_feature_projs is None:
                self.domain_feature_projs = nn.ModuleDict()
            self.domain_feature_projs[name] = MLP(
                config=MLPConfig(
                    input_dim=spec.feature_dim.total_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    bias=config.transformer_block_config.bias,
                ),
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
                name: ReadoutLayer(
                    decoder_config=decoder_config,
                    hidden_dim=self.hidden_dim,
                    condition_dim=config.transformer_block_config.condition_dim,
                )  # type: ignore[arg-type]
                for name, decoder_config in config.domain_decoder_configs.items()  # type: ignore[attr-defined]
            }
        )

        if config.data_specs.conditioning_dims is not None:
            self.conditioner = VectorsConditioner(
                config=config.conditioner_config,  # type: ignore[arg-type]
            )

        geom_dims = config.geometry_conditioning_dims
        if geom_dims is not None and geom_dims != config.data_specs.conditioning_dims:
            self.geometry_conditioner = VectorsConditioner(
                config=config.geometry_conditioner_config,  # type: ignore[arg-type]
            )

        self._geometry_inherits_conditioning: bool = (
            geom_dims is not None
            and config.data_specs.conditioning_dims is not None
            and geom_dims == config.data_specs.conditioning_dims
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

    def _build_domain_segment(
        self,
        name: str,
        anchor_position: torch.Tensor | None,
        query_position: torch.Tensor | None,
        anchor_feature: torch.Tensor | None,
        query_feature: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the physics-block input segment and combined positions for a single domain.

        Concatenates ``[anchor | query]`` along the token dim. Each segment is
        ``bias(pos_embed(pos)) + (proj(features) or 0)``; the returned position
        is the concatenation of anchor and query positions in token order.
        """
        if anchor_position is None and query_position is None:
            raise ValueError(f"domain {name!r} has neither anchor nor query positions")

        proj: torch.nn.Module | None = (
            self.domain_feature_projs[name] if self.domain_feature_projs and name in self.domain_feature_projs else None
        )

        def _segment(pos: torch.Tensor, feat: torch.Tensor | None) -> torch.Tensor:
            if pos.ndim != 3:
                raise ValueError(f"Position tensor for domain '{name}' must be 3-dimensional, got {pos.ndim}.")
            seg: torch.Tensor = self.domain_biases[name](self.pos_embed(pos))
            if proj is not None and feat is not None:
                seg = seg + proj(feat)
            return seg

        seg_parts: list[torch.Tensor] = []
        pos_parts: list[torch.Tensor] = []
        if anchor_position is not None:
            seg_parts.append(_segment(anchor_position, anchor_feature))
            pos_parts.append(anchor_position)
        if query_position is not None:
            seg_parts.append(_segment(query_position, query_feature))
            pos_parts.append(query_position)

        segment = torch.cat(seg_parts, dim=1) if len(seg_parts) > 1 else seg_parts[0]
        position = torch.cat(pos_parts, dim=1) if len(pos_parts) > 1 else pos_parts[0]
        return segment, position

    def build_physics_input(
        self,
        domain_anchor_positions: dict[str, torch.Tensor] | None = None,
        domain_query_positions: dict[str, torch.Tensor] | None = None,
        domain_anchor_features: dict[str, torch.Tensor] | None = None,
        domain_query_features: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Build the physics-block input tensor and combined per-domain positions.

        Each per-domain segment is ``[anchors | queries]`` with positional
        biases plus projected features (when ``data_specs.domains[name].feature_dim``
        was set on the config). Domains are concatenated in ``self.domain_names``
        order.

        Returns:
            Tuple of (x_physics, physics_positions). ``x_physics`` has shape
            ``(B, total_tokens, hidden_dim)``. ``physics_positions`` maps each
            domain name to its concatenated ``[anchors | queries]`` positions
            and can be passed directly to :meth:`create_rope_frequencies`.
        """
        domain_anchor_positions = domain_anchor_positions or {}
        domain_query_positions = domain_query_positions or {}
        domain_anchor_features = domain_anchor_features or {}
        domain_query_features = domain_query_features or {}

        segments: list[torch.Tensor] = []
        physics_positions: dict[str, torch.Tensor] = {}
        for name in self.domain_names:
            segment, position = self._build_domain_segment(
                name,
                anchor_position=domain_anchor_positions.get(name),
                query_position=domain_query_positions.get(name),
                anchor_feature=domain_anchor_features.get(name),
                query_feature=domain_query_features.get(name),
            )
            segments.append(segment)
            physics_positions[name] = position
        return torch.cat(segments, dim=1), physics_positions

    def physics_blocks_forward(
        self,
        x_physics: torch.Tensor,
        geometry_encoding: torch.Tensor | None,
        physics_token_specs: list[TokenSpec],
        physics_attn_kwargs: dict[str, Any],
        physics_perceiver_attn_kwargs: dict[str, Any],
        condition: torch.Tensor | None,
        physics_blocks_cache: list[LayerCache | None] | None = None,
    ) -> tuple[torch.Tensor, list[LayerCache | None]]:
        """Run the physics-block stack on a pre-built input tensor.

        Perceiver blocks always re-project K/V from ``geometry_encoding`` and
        contribute ``None`` to the returned cache; only transformer blocks
        cache their anchor self-attention K/V.
        """
        if physics_blocks_cache is not None:
            assert len(physics_blocks_cache) == len(self.physics_blocks), (
                f"physics_blocks_cache length ({len(physics_blocks_cache)}) must match "
                f"number of physics blocks ({len(self.physics_blocks)})"
            )

        new_physics_cache: list[LayerCache | None] = []
        for i, block in enumerate(self.physics_blocks):
            if isinstance(block, TransformerBlock | UntiedTransformerBlock):
                x_physics, block_cache = block(
                    x_physics,
                    attn_kwargs=dict(
                        token_specs=physics_token_specs,
                        kv_cache=physics_blocks_cache[i] if physics_blocks_cache else None,
                        **physics_attn_kwargs,
                    ),
                    condition=condition,
                )
                new_physics_cache.append(block_cache)
            elif isinstance(block, PerceiverBlock):
                perceiver_attn_kwargs: dict[str, Any] = dict(**physics_perceiver_attn_kwargs)
                if isinstance(block, UntiedPerceiverBlock):
                    perceiver_attn_kwargs["token_specs"] = physics_token_specs
                x_physics, _ = block(
                    q=x_physics,
                    kv=geometry_encoding,
                    attn_kwargs=perceiver_attn_kwargs,
                    condition=condition,
                )
                new_physics_cache.append(None)
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

        return self.domain_decoder_projections[domain_name](x, condition=condition), new_cache

    def decoder_blocks_forward(
        self,
        x_physics: torch.Tensor,
        physics_token_specs: list[TokenSpec],
        per_domain_token_specs: dict[str, list[TokenSpec]],
        decoder_attn_kwargs: dict[str, dict[str, Any]],
        condition: torch.Tensor | None,
        decoders_cache: dict[str, list[LayerCache]] | None = None,
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

            expected_size = sum(spec.size for spec in per_domain_token_specs[name] if spec.size is not None)
            if x_domain.size(1) != expected_size:
                raise ValueError(
                    f"{name} tensor size ({x_domain.size(1)}) does not match expected size ({expected_size})."
                )

            preds, new_cache = self._decode_domain(
                x_domain,
                name,
                per_domain_token_specs[name],
                decoder_attn_kwargs[name],
                condition,
                cache=decoders_cache.get(name) if decoders_cache else None,
            )
            if preds is not None:
                domain_predictions[name] = preds
            new_domain_caches[name] = new_cache

        return domain_predictions, new_domain_caches

    def create_rope_frequencies(
        self,
        physics_positions: dict[str, torch.Tensor],
        geometry_position: torch.Tensor | None = None,
        geometry_supernode_idx: torch.Tensor | None = None,
        geometry_rope: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], torch.Tensor | None]:
        """Create RoPE frequencies for all relevant positions.

        Args:
            physics_positions: Per-domain combined ``[anchors | queries]``
                positions, as returned by :meth:`build_physics_input`.
            geometry_position: Geometry mesh coordinates (optional).
            geometry_supernode_idx: Geometry supernode indices (optional).
            geometry_rope: Precomputed geometry-supernode RoPE. When provided,
                bypasses ``geometry_position`` / ``geometry_supernode_idx`` for
                the perceiver ``k_freqs`` (needed in queries-only mode where
                geometry inputs aren't available).

        Returns:
            Tuple of (geometry_attn_kwargs, decoder_attn_kwargs,
            physics_perceiver_attn_kwargs, physics_attn_kwargs, geometry_rope).
            ``geometry_rope`` is the rope tensor used / computed (or ``None``
            when there's no geometry branch).
        """
        first_pos = next(iter(physics_positions.values()))
        batch_size = first_pos.size(0)

        geometry_attn_kwargs: dict[str, Any] = {}
        physics_perceiver_attn_kwargs: dict[str, Any] = {}
        physics_attn_kwargs: dict[str, Any] = {}

        if geometry_rope is None and geometry_position is not None and geometry_supernode_idx is not None:
            geometry_rope = self.rope(geometry_position[geometry_supernode_idx].unsqueeze(0))
            channels = geometry_rope.shape[-1]
            geometry_rope = geometry_rope.view(batch_size, -1, channels)
        if geometry_rope is not None:
            geometry_attn_kwargs["freqs"] = geometry_rope
            physics_perceiver_attn_kwargs["k_freqs"] = geometry_rope

        # Per-domain rope + concatenated physics rope
        domain_rope = {name: self.rope(physics_positions[name]) for name in self.domain_names}
        rope_all = torch.cat([domain_rope[name] for name in self.domain_names], dim=1)

        physics_perceiver_attn_kwargs["q_freqs"] = rope_all
        physics_attn_kwargs["freqs"] = rope_all

        decoder_attn_kwargs = {name: {"freqs": domain_rope[name]} for name in self.domain_names}

        return (
            geometry_attn_kwargs,
            decoder_attn_kwargs,
            physics_perceiver_attn_kwargs,
            physics_attn_kwargs,
            geometry_rope,
        )

    def forward(
        self,
        # geometry
        geometry_position: torch.Tensor | None = None,
        geometry_supernode_idx: torch.Tensor | None = None,
        geometry_batch_idx: torch.Tensor | None = None,
        # domain positions
        domain_anchor_positions: dict[str, Tensor] | None = None,
        domain_query_positions: dict[str, Tensor] | None = None,
        # domain features (per-token inputs in addition to positions)
        domain_anchor_features: dict[str, Tensor] | None = None,
        domain_query_features: dict[str, Tensor] | None = None,
        conditioning_inputs: dict[str, Tensor] | None = None,
        geometry_conditioning_inputs: dict[str, Tensor] | None = None,
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
                domain_anchor_features={"surface": surface_features, "volume": volume_features},
                domain_query_features={"surface": query_features},
                conditioning_inputs={"geometry_design_parameters": design_params},
            )

        Args:
            geometry_position: Coordinates of the geometry mesh. Tensor of shape (B * N_geometry, D_pos).
            geometry_supernode_idx: Supernode indices for the geometry points.
            geometry_batch_idx: Batch indices for the geometry points.
            domain_anchor_positions: Per-domain anchor positions, e.g. ``{"surface": (B, N, D), "volume": (B, M, D)}``.
            domain_query_positions: Per-domain query positions (optional).
            domain_anchor_features: Per-domain anchor input features (optional), matching the shape of ``domain_anchor_positions``.
            domain_query_features: Per-domain query input features (optional), matching the shape of ``domain_query_positions``.
            conditioning_inputs: Conditioning tensors for physics + decoder blocks,
                e.g. ``{"geometry_design_parameters": (B, D)}``.
            geometry_conditioning_inputs: Conditioning tensors for the geometry
                branch. When ``None`` and ``conditioning_inputs`` is set, the
                geometry branch automatically reuses ``conditioning_inputs``
                if the configured ``geometry_conditioning_dims`` matches
                ``data_specs.conditioning_dims`` (the common case). Pass an
                explicit dict to feed a different conditioning to geometry,
                or leave it ``None`` when the geometry branch is unconditioned.
            kv_cache: KV cache from a previous forward call.

        Returns:
            Tuple of (predictions, kv_cache).
        """
        domain_anchor_positions = domain_anchor_positions or {}
        domain_query_positions = domain_query_positions or {}
        kv_cache = kv_cache if kv_cache is not None else {}

        # The "physics_blocks" cache, when present, holds anchor self-attention
        # K/V — i.e. anchors are already encoded and this call is decode-only.
        # When absent, anchors are re-encoded (so anchor features can change).
        anchors_cached = "physics_blocks" in kv_cache
        has_anchors = bool(domain_anchor_positions)
        use_cached_geometry = "geometry_encoding" in kv_cache

        if has_anchors == anchors_cached:
            raise ValueError(
                "Either domain anchor positions must be provided (no anchor KV cache) "
                "or a kv_cache containing 'physics_blocks' must be provided, but not both."
            )

        if anchors_cached:
            assert geometry_position is None, "geometry_position must be None when reusing anchor KV cache"
            assert geometry_supernode_idx is None, "geometry_supernode_idx must be None when reusing anchor KV cache"
            assert geometry_batch_idx is None, "geometry_batch_idx must be None when reusing anchor KV cache"
            assert domain_query_positions, (
                "At least one domain query position must be provided when reusing anchor KV cache"
            )
        elif self.use_geometry_branch and not use_cached_geometry:
            assert geometry_position is not None, "geometry_position is required when using geometry branch"
            assert geometry_supernode_idx is not None, "geometry_supernode_idx is required when using geometry branch"
            assert geometry_batch_idx is not None, "geometry_batch_idx is required when using geometry branch"

        condition = None
        if conditioning_inputs is not None:
            condition = self.conditioner(**conditioning_inputs)

        geometry_condition = None
        if geometry_conditioning_inputs is not None:
            geometry_condition = self.geometry_conditioner(**geometry_conditioning_inputs)
        elif self._geometry_inherits_conditioning:
            geometry_condition = condition

        # Create token specifications
        physics_token_specs, per_domain_token_specs = self._create_all_token_specs(
            domain_anchor_positions=domain_anchor_positions,
            domain_query_positions=domain_query_positions,
            use_cached_kv=anchors_cached,
        )

        # Physics blocks: build the per-domain input tensor, then run the block stack.
        x_physics, physics_positions = self.build_physics_input(
            domain_anchor_positions=domain_anchor_positions,
            domain_query_positions=domain_query_positions,
            domain_anchor_features=domain_anchor_features,
            domain_query_features=domain_query_features,
        )

        # RoPE frequencies. ``geometry_rope`` is reused from the cache when
        # available so queries-only inference works without geometry_position.
        geometry_attn_kwargs, decoder_attn_kwargs, physics_perceiver_attn_kwargs, physics_attn_kwargs, geometry_rope = (
            self.create_rope_frequencies(
                physics_positions=physics_positions,
                geometry_position=geometry_position,
                geometry_supernode_idx=geometry_supernode_idx,
                geometry_rope=kv_cache.get("geometry_rope"),
            )
        )

        # Geometry branch: reuse cached encoding when available, otherwise encode.
        geometry_encoding: Tensor | None = kv_cache.get("geometry_encoding")
        if geometry_encoding is None and self.use_geometry_branch:
            assert geometry_position is not None
            assert geometry_supernode_idx is not None
            assert geometry_batch_idx is not None
            geometry_encoding = self.geometry_branch_forward(
                geometry_position=geometry_position,
                geometry_supernode_idx=geometry_supernode_idx,
                geometry_batch_idx=geometry_batch_idx,
                condition=geometry_condition,
                geometry_attn_kwargs=geometry_attn_kwargs,
            )

        x_physics, new_physics_cache = self.physics_blocks_forward(
            x_physics=x_physics,
            geometry_encoding=geometry_encoding,
            physics_token_specs=physics_token_specs,
            physics_attn_kwargs=physics_attn_kwargs,
            physics_perceiver_attn_kwargs=physics_perceiver_attn_kwargs,
            condition=condition,
            physics_blocks_cache=kv_cache.get("physics_blocks"),
        )

        # Decoder blocks
        domain_predictions, new_domain_caches = self.decoder_blocks_forward(
            x_physics=x_physics,
            physics_token_specs=physics_token_specs,
            per_domain_token_specs=per_domain_token_specs,
            decoder_attn_kwargs=decoder_attn_kwargs,
            condition=condition,
            decoders_cache=kv_cache.get("decoders"),
        )

        # Slice predictions into named fields
        predictions: dict[str, Tensor] = {}
        for name, preds in domain_predictions.items():
            num_anchors = domain_anchor_positions[name].size(1) if name in domain_anchor_positions else 0
            predictions.update(self._slice_predictions(preds, name, num_anchors, anchors_cached))

        # Build / refresh the cache. When anchors were re-encoded, replace the
        # physics_blocks and decoders entries; the geometry_encoding entry (if
        # cached) is preserved by passthrough.
        new_kv_cache: ModelKVCache = {}
        if geometry_encoding is not None:
            new_kv_cache["geometry_encoding"] = geometry_encoding
        if geometry_rope is not None:
            new_kv_cache["geometry_rope"] = geometry_rope
        if anchors_cached:
            new_kv_cache["physics_blocks"] = kv_cache["physics_blocks"]
            if "decoders" in kv_cache:
                new_kv_cache["decoders"] = kv_cache["decoders"]
        else:
            new_kv_cache["physics_blocks"] = new_physics_cache
            new_kv_cache["decoders"] = new_domain_caches

        return predictions, new_kv_cache
