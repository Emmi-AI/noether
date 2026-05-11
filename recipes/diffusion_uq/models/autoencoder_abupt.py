#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import ConfigDict, Field, model_validator
from torch import Tensor, nn

from noether.core.models import Model
from noether.core.schemas.models.ab_upt import AnchorBranchedUPTConfig
from noether.core.schemas.modules.attention import TokenSpec
from noether.modeling.models.ab_upt import AnchoredBranchedUPT
from noether.modeling.modules.blocks import PerceiverBlock, TransformerBlock


class ABUPTAutoencoderConfig(AnchorBranchedUPTConfig):
    """ab-upt config extended for autoencoder with latent bottleneck.

    inherits the full ab-upt backbone config (geometry encoder, physics blocks,
    per-domain decoders) and adds latent bottleneck + field projection settings.

    field-projection input dims for each domain are derived from
    ``data_specs.domains[name].output_dims.total_dim`` — the AE re-encodes its
    own training targets, so the per-domain field dim equals the per-domain
    output dim.
    """

    model_config = ConfigDict(extra="forbid")

    latent_dim: int = Field(256, ge=1)
    query_ratio: float = Field(0.0, ge=0.0, lt=1.0)
    """Fraction of anchor points used as decode-only queries during training.
    0.0 = standard AE (encode all, decode all).
    >0  = encode (1-ratio) of anchors, decode at ALL anchors — forces position-independent latent."""

    bottleneck_mode: Literal["perceiver", "sampled"] | None = None
    """Token-count bottleneck mode (None = no bottleneck, latent is 1:1 with anchors):
    - "perceiver": K learned-parameter query tokens, no spatial position
    - "sampled":   K positions uniformly sampled from anchors per forward
    For sampled, latent tokens carry positional info → enables RoPE / 2D-3D
    diffusion tricks downstream."""

    latent_num_tokens: dict[str, int] | None = None
    """Per-domain ``K`` latent token counts, e.g. ``{"surface": 512, "volume": 512}``.
    Required when ``bottleneck_mode`` is set; must contain a key for every
    domain in ``data_specs.domains``."""

    bottleneck_num_heads: int = Field(4, ge=1)
    """Attention heads for the bottleneck cross-attn blocks."""

    @model_validator(mode="after")
    def _validate_bottleneck(self) -> ABUPTAutoencoderConfig:
        if self.bottleneck_mode is None:
            return self
        if self.latent_num_tokens is None:
            raise ValueError(f"{self.bottleneck_mode!r} mode requires latent_num_tokens")
        missing = set(self.data_specs.domains.keys()) - set(self.latent_num_tokens.keys())
        if missing:
            raise ValueError(
                f"latent_num_tokens is missing entries for domain(s) {sorted(missing)}; "
                f"got keys {sorted(self.latent_num_tokens.keys())}"
            )
        for name, k in self.latent_num_tokens.items():
            if k < 1:
                raise ValueError(f"latent_num_tokens[{name!r}] must be >= 1, got {k}")
        return self


class CrossAttnBlock(nn.Module):
    """Pre-LN cross-attention block with MLP. Used as the token-count bottleneck."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False)
        q = q + attn_out
        q = q + self.mlp(self.norm_mlp(q))
        return q


class ABUPTAutoencoder(Model):
    """AB-UPT autoencoder with explicit latent bottleneck.

    Encode: geometry_branch -> physics_blocks (+ field injection) -> downproj -> latent.
    Decode: upproj -> decoder_blocks -> field predictions.

    Shares the AB-UPT backbone with DiffusionABUPT. Field injection into
    domain anchor embeddings lets the latent capture field variation, not
    just geometry. Optional token-count bottleneck makes the latent
    resolution-independent (decode at arbitrary positions from a fixed latent).

    Domain-generic: iterates over ``backbone.domain_names`` (typically
    ``("surface", "volume")``) and stores per-domain submodules in
    ``ModuleDict``s. Field-projection input dims are derived from
    ``data_specs.domains[name].output_dims.total_dim`` since the AE injects
    its training targets back into the encoder.
    """

    def __init__(self, model_config: ABUPTAutoencoderConfig, **kwargs):
        super().__init__(model_config=model_config, **kwargs)
        cfg = model_config
        hidden_dim = cfg.hidden_dim

        self.backbone = AnchoredBranchedUPT(config=cfg)
        self.data_specs = cfg.data_specs
        self.domain_names: list[str] = list(self.backbone.domain_names)

        # Per-domain field projection MLPs (input dim = total output dim of the
        # domain, since the AE re-encodes its own targets).
        self.domain_field_projs = nn.ModuleDict()
        for name in self.domain_names:
            field_dim = cfg.data_specs.domains[name].output_dims.total_dim
            if field_dim > 0:
                self.domain_field_projs[name] = nn.Sequential(
                    nn.Linear(field_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )

        self.downproj = nn.Linear(hidden_dim, cfg.latent_dim)
        self.upproj = nn.Linear(cfg.latent_dim, hidden_dim)

        self.bottleneck_mode = cfg.bottleneck_mode  # "perceiver" | "sampled" | None
        self.use_token_bottleneck = self.bottleneck_mode is not None
        if self.use_token_bottleneck:
            self.encode_bottlenecks = nn.ModuleDict(
                {name: CrossAttnBlock(hidden_dim, cfg.bottleneck_num_heads) for name in self.domain_names}
            )
            self.decode_expanders = nn.ModuleDict(
                {name: CrossAttnBlock(hidden_dim, cfg.bottleneck_num_heads) for name in self.domain_names}
            )

            if self.bottleneck_mode == "perceiver":
                self.latent_qs = nn.ParameterDict(
                    {
                        name: nn.Parameter(torch.randn(1, cfg.latent_num_tokens[name], hidden_dim) * 0.02)
                        for name in self.domain_names
                    }
                )

    @property
    def latent_num_tokens(self) -> dict[str, int]:
        return self.model_config.latent_num_tokens or {}

    def _physics_blocks_with_fields(
        self,
        domain_positions_all: dict[str, Tensor],
        domain_anchor_counts: dict[str, int],
        domain_fields: dict[str, Tensor | None],
        geometry_encoding: Tensor | None,
        physics_token_specs: list[TokenSpec],
        physics_attn_kwargs: dict[str, Any],
        physics_perceiver_attn_kwargs: dict[str, Any],
    ) -> Tensor:
        """Replicate ``backbone.physics_blocks_forward`` with field injection.

        Field projections are added to the position embeddings for anchor
        tokens before the physics blocks; query positions (decode-only) get
        zeros so the field branch only sees encode-set fields.
        """
        bb = self.backbone

        first_pos = next(iter(domain_positions_all.values()))
        batch_size = first_pos.size(0)
        total_tokens = sum(p.size(1) for p in domain_positions_all.values())
        x_physics = torch.empty(batch_size, total_tokens, bb.hidden_dim, device=first_pos.device, dtype=first_pos.dtype)

        start = 0
        for name in bb.domain_names:
            pos = domain_positions_all[name]
            emb = bb.domain_biases[name](bb.pos_embed(pos))

            fields = domain_fields.get(name)
            if fields is not None and name in self.domain_field_projs:
                field_emb = self.domain_field_projs[name](fields)
                n_total = pos.size(1)
                n_anchor = domain_anchor_counts.get(name, n_total)
                if n_anchor < n_total:
                    pad = torch.zeros(
                        batch_size,
                        n_total - n_anchor,
                        field_emb.size(-1),
                        device=field_emb.device,
                        dtype=field_emb.dtype,
                    )
                    field_emb = torch.cat([field_emb, pad], dim=1)
                emb = emb + field_emb

            end = start + emb.size(1)
            x_physics[:, start:end, :] = emb
            start = end

        for block in bb.physics_blocks:
            if isinstance(block, TransformerBlock):
                x_physics, _ = block(
                    x_physics,
                    attn_kwargs=dict(token_specs=physics_token_specs, **physics_attn_kwargs),
                    condition=None,
                )
            elif isinstance(block, PerceiverBlock):
                x_physics, _ = block(
                    q=x_physics,
                    kv=geometry_encoding,
                    attn_kwargs=dict(**physics_perceiver_attn_kwargs),
                    condition=None,
                )
            else:
                raise NotImplementedError(f"Unsupported physics block type: {type(block)}")
        return x_physics

    def _build_bottleneck_queries_for_domain(
        self,
        name: str,
        anchor_position: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        """Build encoder bottleneck queries and their 3D positions for one domain.

        Returns:
            ``(Q, super_pos)``. ``super_pos`` is ``None`` for the perceiver mode
            (learned queries have no spatial position).
        """
        bb = self.backbone
        B = anchor_position.shape[0]

        if self.bottleneck_mode == "perceiver":
            return self.latent_qs[name].expand(B, -1, -1), None

        # sampled mode: subsample anchor positions to form "superanchors".
        K = self.latent_num_tokens[name]
        N = anchor_position.shape[1]
        idx = torch.randint(N, (K,), device=anchor_position.device)
        super_pos = anchor_position[:, idx]
        Q = bb.domain_biases[name](bb.pos_embed(super_pos))
        return Q, super_pos

    def encode(
        self,
        geometry_position: Tensor | None,
        geometry_supernode_idx: Tensor | None,
        geometry_batch_idx: Tensor | None,
        domain_anchor_positions: dict[str, Tensor],
        domain_fields: dict[str, Tensor | None] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor | None]]:
        """Encode inputs to latent tokens.

        Args:
            domain_anchor_positions: ``{domain_name: (B, N_d, position_dim)}``.
            domain_fields: ``{domain_name: (B, N_d, field_dim_d)}`` of training targets
                injected into the encoder. ``None`` per domain to skip injection.

        Returns:
            ``(latents, domain_super_positions)``. ``latents`` has shape
            ``(B, sum(K_d), latent_dim)`` when bottleneck is on, otherwise
            ``(B, sum(N_d), latent_dim)``. ``domain_super_positions`` maps
            domain name → super-token positions (``None`` for ``perceiver`` /
            no-bottleneck modes).
        """
        bb = self.backbone
        domain_fields = domain_fields or {}

        physics_token_specs, _ = bb._create_all_token_specs(
            domain_anchor_positions=domain_anchor_positions,
            domain_query_positions={},
        )

        domain_positions_all = dict(domain_anchor_positions)
        domain_anchor_counts = {name: pos.size(1) for name, pos in domain_anchor_positions.items()}

        (
            geometry_attn_kwargs,
            _,
            physics_perceiver_attn_kwargs,
            physics_attn_kwargs,
        ) = bb.create_rope_frequencies(
            domain_anchor_positions=domain_anchor_positions,
            domain_query_positions={},
            geometry_position=geometry_position,
            geometry_supernode_idx=geometry_supernode_idx,
        )

        geometry_encoding = None
        if bb.use_geometry_branch:
            geometry_encoding = bb.geometry_branch_forward(
                geometry_position=geometry_position,
                geometry_supernode_idx=geometry_supernode_idx,
                geometry_batch_idx=geometry_batch_idx,
                condition=None,
                geometry_attn_kwargs=geometry_attn_kwargs,
            )

        x_physics = self._physics_blocks_with_fields(
            domain_positions_all=domain_positions_all,
            domain_anchor_counts=domain_anchor_counts,
            domain_fields=domain_fields,
            geometry_encoding=geometry_encoding,
            physics_token_specs=physics_token_specs,
            physics_attn_kwargs=physics_attn_kwargs,
            physics_perceiver_attn_kwargs=physics_perceiver_attn_kwargs,
        )

        domain_super_positions: dict[str, Tensor | None] = dict.fromkeys(bb.domain_names)
        if self.use_token_bottleneck:
            x_per_domain = bb._split_domain_tensors(x_physics, physics_token_specs)
            z_per_domain: dict[str, Tensor] = {}
            for name in bb.domain_names:
                Q, sp = self._build_bottleneck_queries_for_domain(name, domain_anchor_positions[name])
                z_per_domain[name] = self.encode_bottlenecks[name](Q, x_per_domain[name])
                domain_super_positions[name] = sp
            x_physics = torch.cat([z_per_domain[name] for name in bb.domain_names], dim=1)

        return self.downproj(x_physics), domain_super_positions

    def decode(
        self,
        latents: Tensor,
        domain_anchor_positions: dict[str, Tensor],
        domain_query_positions: dict[str, Tensor] | None = None,
        geometry_position: Tensor | None = None,
        geometry_supernode_idx: Tensor | None = None,
        domain_super_positions: dict[str, Tensor | None] | None = None,
    ) -> dict[str, Tensor]:
        """Decode latent tokens to per-field predictions.

        Without bottleneck: ``latents`` are 1:1 with anchors; query positions
        are extended by position embeddings before the decoder.

        With bottleneck: ``latents`` are ``sum(K_d)`` tokens. Anchor and query
        positions are projected to position embeddings and cross-attend to the
        per-domain latent slice (``decode_expanders[name]``). For ``sampled``
        mode the super-token 3D positions are added back to the latent side.

        Returns:
            Dict keyed by ``{domain}_{field}`` and ``query_{domain}_{field}``.
        """
        bb = self.backbone
        domain_query_positions = domain_query_positions or {}
        domain_super_positions = domain_super_positions or {}

        x_latent = self.upproj(latents)

        # Concatenated all-positions per domain (anchors + queries).
        domain_positions_all: dict[str, Tensor] = {}
        for name in bb.domain_names:
            anchor = domain_anchor_positions.get(name)
            query = domain_query_positions.get(name)
            if anchor is not None and query is not None:
                domain_positions_all[name] = torch.cat([anchor, query], dim=1)
            elif anchor is not None:
                domain_positions_all[name] = anchor
            elif query is not None:
                domain_positions_all[name] = query
            else:
                ref = next(iter((domain_anchor_positions or domain_query_positions).values()))
                domain_positions_all[name] = ref[:, :0]

        physics_token_specs, per_domain_token_specs = bb._create_all_token_specs(
            domain_anchor_positions=domain_anchor_positions,
            domain_query_positions=domain_query_positions,
        )

        _, decoder_attn_kwargs, _, _ = bb.create_rope_frequencies(
            domain_anchor_positions=domain_anchor_positions,
            domain_query_positions=domain_query_positions,
            geometry_position=geometry_position,
            geometry_supernode_idx=geometry_supernode_idx,
        )

        if self.use_token_bottleneck:
            # Slice the latent tensor into per-domain blocks of size K_d.
            z_per_domain: dict[str, Tensor] = {}
            offset = 0
            for name in bb.domain_names:
                K = self.latent_num_tokens[name]
                z_per_domain[name] = x_latent[:, offset : offset + K]
                offset += K

            if self.bottleneck_mode == "sampled":
                for name in bb.domain_names:
                    sp = domain_super_positions.get(name)
                    if sp is None:
                        raise ValueError(
                            f"bottleneck_mode={self.bottleneck_mode!r} decode requires "
                            f"super_positions for domain '{name}'"
                        )
                    z_per_domain[name] = z_per_domain[name] + bb.domain_biases[name](bb.pos_embed(sp))

            x_per_domain: dict[str, Tensor] = {}
            for name in bb.domain_names:
                pos_emb = bb.domain_biases[name](bb.pos_embed(domain_positions_all[name]))
                x_per_domain[name] = self.decode_expanders[name](pos_emb, z_per_domain[name])
            x_physics = torch.cat([x_per_domain[name] for name in bb.domain_names], dim=1)
        else:
            # No bottleneck: latents are 1:1 with anchors. Extend each domain
            # with position embeddings for any decode-only queries.
            anchor_token_specs, _ = bb._create_all_token_specs(
                domain_anchor_positions=domain_anchor_positions,
                domain_query_positions={},
            )
            x_per_domain_anchors = bb._split_domain_tensors(x_latent, anchor_token_specs)
            x_per_domain = {}
            for name in bb.domain_names:
                x_d = x_per_domain_anchors[name]
                query_pos = domain_query_positions.get(name)
                if query_pos is not None and query_pos.size(1) > 0:
                    q_emb = bb.domain_biases[name](bb.pos_embed(query_pos))
                    x_d = torch.cat([x_d, q_emb], dim=1)
                x_per_domain[name] = x_d
            x_physics = torch.cat([x_per_domain[name] for name in bb.domain_names], dim=1)

        domain_predictions, _ = bb.decoder_blocks_forward(
            x_physics=x_physics,
            physics_token_specs=physics_token_specs,
            per_domain_token_specs=per_domain_token_specs,
            decoder_attn_kwargs=decoder_attn_kwargs,
            condition=None,
            domain_anchor_positions=domain_anchor_positions,
            domain_query_positions=domain_query_positions,
        )

        result: dict[str, Tensor] = {}
        for name, preds in domain_predictions.items():
            num_anchors = domain_anchor_positions[name].size(1) if name in domain_anchor_positions else 0
            result.update(bb._slice_predictions(preds, name, num_anchors))
        return result

    def _build_domain_fields(self, kwargs: dict[str, Tensor]) -> dict[str, Tensor | None]:
        """Concatenate per-field training targets into a per-domain field tensor.

        Reads ``{domain}_{field}_target`` keys (matching the trainer's batch
        keys) and concatenates them in ``data_specs`` field order. Returns
        ``None`` per domain when the AE has no field projection for it or no
        targets are present.
        """
        domain_fields: dict[str, Tensor | None] = {}
        for name in self.domain_names:
            if name not in self.domain_field_projs:
                domain_fields[name] = None
                continue
            spec = self.data_specs.domains[name]
            parts = []
            for field_name in spec.output_dims.keys():
                key = f"{name}_{field_name}_target"
                if key in kwargs and kwargs[key] is not None:
                    parts.append(kwargs[key])
            domain_fields[name] = torch.cat(parts, dim=-1) if parts else None
        return domain_fields

    @staticmethod
    def _unpermute_predictions(
        preds: dict[str, Tensor],
        domain_perms: dict[str, Tensor],
        domain_n_enc: dict[str, int],
    ) -> dict[str, Tensor]:
        """Reassemble anchor+query predictions into the original point order.

        Generic over ``domain_names``: for each domain, looks up the
        permutation ``perm`` (length ``N``) and the encode-set size ``n_enc``,
        then scatters anchor predictions back to ``perm[:n_enc]`` and query
        predictions to ``perm[n_enc:]``.
        """
        result: dict[str, Tensor] = {}

        for key, val in preds.items():
            is_query = key.startswith("query_")
            stripped = key.removeprefix("query_") if is_query else key

            domain = next((d for d in domain_perms if stripped.startswith(f"{d}_")), None)
            if domain is None:
                result[key] = val
                continue

            perm = domain_perms[domain]
            n_enc = domain_n_enc[domain]
            enc_idx, qry_idx = perm[:n_enc], perm[n_enc:]

            if stripped not in result:
                B = val.shape[0]
                N = len(perm)
                result[stripped] = torch.empty(B, N, val.shape[-1], device=val.device, dtype=val.dtype)

            if is_query:
                result[stripped][:, qry_idx] = val
            else:
                result[stripped][:, enc_idx] = val

        return result

    def forward(
        self,
        geometry_position: Tensor | None = None,
        geometry_supernode_idx: Tensor | None = None,
        geometry_batch_idx: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Autoencoder forward: encode fields -> latent -> decode back to fields.

        Reads ``{domain}_anchor_position`` and ``{domain}_{field}_target`` keys
        from ``kwargs`` (matching the trainer's batch contract). When
        ``query_ratio > 0`` and training, randomly splits each domain's
        anchors into an encode-set and query-set; encodes at the encode-set
        with field values, decodes at all positions. Forces the latent to be
        a position-independent field encoding.
        """
        domain_anchor_positions: dict[str, Tensor] = {}
        for name in self.domain_names:
            key = f"{name}_anchor_position"
            if key in kwargs and kwargs[key] is not None:
                domain_anchor_positions[name] = kwargs[key]

        domain_fields = self._build_domain_fields(kwargs)

        query_ratio = self.model_config.query_ratio
        use_queries = query_ratio > 0 and self.training

        if use_queries:
            domain_perms: dict[str, Tensor] = {}
            domain_n_enc: dict[str, int] = {}
            domain_enc_anchors: dict[str, Tensor] = {}
            domain_query_anchors: dict[str, Tensor] = {}
            domain_enc_fields: dict[str, Tensor | None] = {}

            for name, pos in domain_anchor_positions.items():
                N = pos.shape[1]
                n_enc = max(1, int(N * (1 - query_ratio)))
                perm = torch.randperm(N, device=pos.device)
                enc_idx = perm[:n_enc]
                qry_idx = perm[n_enc:]
                domain_perms[name] = perm
                domain_n_enc[name] = n_enc
                domain_enc_anchors[name] = pos[:, enc_idx]
                if len(qry_idx) > 0:
                    domain_query_anchors[name] = pos[:, qry_idx]
                fields = domain_fields.get(name)
                domain_enc_fields[name] = fields[:, enc_idx] if fields is not None else None

            latents, super_positions = self.encode(
                geometry_position=geometry_position,
                geometry_supernode_idx=geometry_supernode_idx,
                geometry_batch_idx=geometry_batch_idx,
                domain_anchor_positions=domain_enc_anchors,
                domain_fields=domain_enc_fields,
            )

            preds = self.decode(
                latents=latents,
                domain_anchor_positions=domain_enc_anchors,
                domain_query_positions=domain_query_anchors or None,
                geometry_position=geometry_position,
                geometry_supernode_idx=geometry_supernode_idx,
                domain_super_positions=super_positions,
            )

            return self._unpermute_predictions(preds, domain_perms, domain_n_enc)

        latents, super_positions = self.encode(
            geometry_position=geometry_position,
            geometry_supernode_idx=geometry_supernode_idx,
            geometry_batch_idx=geometry_batch_idx,
            domain_anchor_positions=domain_anchor_positions,
            domain_fields=domain_fields,
        )

        return self.decode(
            latents=latents,
            domain_anchor_positions=domain_anchor_positions,
            geometry_position=geometry_position,
            geometry_supernode_idx=geometry_supernode_idx,
            domain_super_positions=super_positions,
        )
