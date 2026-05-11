#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections import OrderedDict

from pydantic import ConfigDict, model_validator
from torch import Tensor

from noether.core.models import Model
from noether.core.schemas.dataset import FieldDimSpec
from noether.core.schemas.models.ab_upt import AnchorBranchedUPTConfig
from noether.modeling.models.ab_upt import AnchoredBranchedUPT


class DiffusionABUPTConfig(AnchorBranchedUPTConfig):
    """AB-UPT config extended for data-space diffusion.

    The base AB-UPT now has a built-in :class:`VectorsConditioner` (sin-cos
    embed + MLP per named scalar/vector input) and per-domain feature
    projection MLPs driven by ``data_specs.domains[name].feature_dim``. This
    subclass just wires the diffusion-specific bits on top:

    * ``"timestep": 1`` is merged into ``data_specs.conditioning_dims`` —
      the timestep is a single scalar in ``[0, 1]`` and the backbone's
      conditioner handles the embedding internally, so we don't ship our own
      ``ContinuousSincosEmbed`` or ``time_mlp``. Any conditioning fields the
      user already declared on the dataset (e.g. design parameters) are kept;
      the conditioner just sees one extra named input.
    * ``geometry_conditioning_dims`` is set to the user-supplied
      ``conditioning_dims`` *without* the timestep — geometry is invariant to
      noise level, so the geometry branch shouldn't see it. If the user
      supplied no conditioning, the geometry branch stays fully unconditioned.
    * Each domain's ``feature_dim`` is set to its ``output_dims`` so the
      backbone builds noisy-field projection MLPs and accepts noisy fields
      via ``domain_anchor_features`` / ``domain_query_features``.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def setup_diffusion_conditioning(self) -> DiffusionABUPTConfig:
        """Wire timestep conditioning and per-domain noisy-field projections.

        Runs after the parent's ``set_condition_dim`` /
        ``default_geometry_conditioning`` validators. Merges ``timestep`` into
        any user-supplied ``data_specs.conditioning_dims`` (rather than
        overwriting), routes the *non-timestep* part to
        ``geometry_conditioning_dims`` (geometry stays invariant to noise
        level), and bumps ``transformer_block_config.condition_dim`` so block
        construction sees the conditioning we just declared.

        Idempotent: if ``timestep: 1`` is already present (e.g. the config was
        reloaded from a previous resolved-hp dump), the merge is a no-op and
        we treat the rest of ``conditioning_dims`` as the user-supplied part.
        Any other ``timestep`` value is still an error.
        """
        existing = self.data_specs.conditioning_dims.root if self.data_specs.conditioning_dims else {}
        if "timestep" in existing and existing["timestep"] != 1:
            raise ValueError(
                "'timestep' is reserved for diffusion timestep conditioning with dim=1; "
                f"got {existing['timestep']}. Remove it from data_specs.conditioning_dims."
            )

        if self.geometry_conditioning_dims is not None and "timestep" in self.geometry_conditioning_dims.root:
            del self.geometry_conditioning_dims.root[
                "timestep"
            ]  # just in case the parent validator merged it in already; we want to be sure it's not there for geometry
            if not self.geometry_conditioning_dims.root:
                self.geometry_conditioning_dims = None  # avoid empty FieldDimSpec with no fields

        # ``user_conditioning`` is everything *except* timestep; on reload that's
        # already the case once we strip the merged-in timestep.
        user_root = OrderedDict((k, v) for k, v in existing.items() if k != "timestep")
        user_conditioning: FieldDimSpec | None = FieldDimSpec(root=user_root) if user_root else None
        merged: OrderedDict[str, int] = OrderedDict(user_root)
        merged["timestep"] = 1
        self.data_specs.conditioning_dims = FieldDimSpec(root=merged)
        self.transformer_block_config.condition_dim = (
            self.condition_dim if self.condition_dim is not None else self.hidden_dim
        )  # type: ignore
        # The diffusion model denoises the channels it predicts; declare
        # per-domain feature_dim = output_dims so the backbone builds matching
        # projection MLPs that consume the noisy fields at anchor (and query)
        # positions.
        for spec in self.data_specs.domains.values():
            spec.feature_dim = spec.output_dims.model_copy(deep=True)
        return self


class DiffusionABUPT(Model):
    """AB-UPT backbone for data-space diffusion.

    Thin wrapper that delegates to :class:`AnchoredBranchedUPT` and relies on
    the backbone's built-in features:

    * **Timestep conditioning** via the backbone's ``VectorsConditioner``.
      We pass the raw normalized timestep as ``conditioning_inputs={"timestep": t}``;
      the conditioner does the sin-cos embed + MLP internally. Any extra
      conditioning fields the user declared on the dataset (in
      ``data_specs.conditioning_dims``) are passed through via
      ``conditioning_inputs`` and re-routed to the geometry branch (without
      the timestep).
    * **Noisy-field injection** via per-domain feature MLPs. Each domain's
      ``feature_dim`` is set to its ``output_dims`` in the config, so the
      backbone projects noisy fields to ``hidden_dim`` and adds them to the
      anchor/query position embeddings.
    * **Per-domain readout** via the backbone's ``ReadoutLayer`` (LayerNorm +
      adaLN-zero modulation + linear projection).

    Forward returns ``{name}_anchor_noise`` and (when queries are supplied)
    ``{name}_query_noise`` — each is a flat per-domain noise prediction
    spanning all output channels of that domain.
    """

    def __init__(self, model_config: DiffusionABUPTConfig, **kwargs):
        super().__init__(model_config=model_config, **kwargs)
        self.backbone = AnchoredBranchedUPT(config=model_config)
        self.data_specs = model_config.data_specs
        self.domain_names: list[str] = list(self.backbone.domain_names)

    def forward(
        self,
        timestep: Tensor,
        geometry_position: Tensor | None = None,
        geometry_supernode_idx: Tensor | None = None,
        geometry_batch_idx: Tensor | None = None,
        domain_anchor_positions: dict[str, Tensor] | None = None,
        domain_query_positions: dict[str, Tensor] | None = None,
        domain_anchor_features: dict[str, Tensor] | None = None,
        domain_query_features: dict[str, Tensor] | None = None,
        conditioning_inputs: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Diffusion forward: noise prediction at anchor (and optional query) positions.

        Args:
            timestep: ``(B,)`` or ``(B, 1)`` diffusion timestep / sigma / flow time.
                Reshaped to ``(B, 1)`` before the conditioner; values should be
                roughly in ``[0, 1]`` (or ``[-1, 1]``) — see
                :class:`VectorsConditioner` on input range.
            geometry_position: Geometry mesh coordinates ``(B * N_geometry, D)``.
            geometry_supernode_idx: Supernode indices for the geometry points.
            geometry_batch_idx: Batch indices for the geometry points.
            domain_anchor_positions: Per-domain anchor positions, e.g.
                ``{"surface": (B, N_s, D), "volume": (B, N_v, D)}``.
            domain_query_positions: Per-domain query positions (optional).
            domain_anchor_features: Per-domain noisy fields at anchor positions
                (shape matches ``data_specs.domains[name].output_dims.total_dim``
                in the channel axis).
            domain_query_features: Per-domain noisy fields at query positions
                (optional, only when ``domain_query_positions`` is given).
            conditioning_inputs: Extra conditioning fields declared in
                ``data_specs.conditioning_dims`` (other than ``timestep``).
                Every non-timestep entry must be supplied; the same dict is
                forwarded to the geometry branch (so geometry sees user
                conditioning but never the noise level).

        Returns:
            ``{name}_anchor_noise`` and ``{name}_query_noise`` (the latter only
            when query positions were supplied), each of shape
            ``(B, N, total_output_dims)``.
        """
        cond_dims = self.data_specs.conditioning_dims
        assert cond_dims is not None  # set in DiffusionABUPTConfig.setup_diffusion_conditioning
        full_conditioning: dict[str, Tensor] = {"timestep": timestep.view(-1, 1)}
        geometry_conditioning: dict[str, Tensor] = {}
        for cond_name in cond_dims.keys():
            if cond_name == "timestep":
                continue
            if conditioning_inputs is None or (cond := conditioning_inputs.get(cond_name)) is None:
                raise ValueError(f"missing conditioning input {cond_name!r} declared in data_specs.conditioning_dims")
            full_conditioning[cond_name] = cond
            geometry_conditioning[cond_name] = cond

        predictions, _ = self.backbone(
            geometry_position=geometry_position,
            geometry_supernode_idx=geometry_supernode_idx,
            geometry_batch_idx=geometry_batch_idx,
            domain_anchor_positions=domain_anchor_positions,
            domain_query_positions=domain_query_positions,
            domain_anchor_features=domain_anchor_features,
            domain_query_features=domain_query_features,
            conditioning_inputs=full_conditioning,
            geometry_conditioning_inputs=geometry_conditioning or None,
        )
        return predictions
