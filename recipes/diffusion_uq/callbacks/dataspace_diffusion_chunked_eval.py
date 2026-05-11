#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Chunked full-mesh evaluation callback for data-space AB-UPT diffusion.

Subclasses tutorial's SurfaceVolumeEvaluationMetricsCallback. Same metric /
denormalize / logging machinery — only swaps `_run_model_inference` to route
through `model.sample(schedule, ...)` (FM Euler) and to split the concatenated
anchor-field tensors into the per-field keys expected by `_compute_mode_metrics`.

Intended paired with a `chunked_test` dataset whose pipeline sets
``num_*_anchor_points`` to the full mesh size — chunks are then sliced back
to the training anchor count via `chunk_properties` / `chunk_size`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

import torch
from aero_cfd.callbacks.aero_metrics import AeroMetricsCallback, AeroMetricsCallbackConfig
from models.diffusion_abupt import DiffusionABUPT
from pydantic import Field

from noether.core.schemas.diffusion import AnyDiffusionScheduleConfig, FlowMatchingConfig
from noether.modeling.diffusion import build_schedule
from noether.modeling.diffusion.flow_matching import FlowMatchingSchedule


class DataspaceDiffusionChunkedEvalCallbackConfig(AeroMetricsCallbackConfig):
    """Config for chunked schedule-driven eval on AB-UPT data-space diffusion.

    Inherits chunk_properties / chunk_size / dataset_key / forward_properties
    from the tutorial callback config and adds sampling-specific knobs.
    """

    name: Literal["DataspaceDiffusionChunkedEvalCallback"] = "DataspaceDiffusionChunkedEvalCallback"
    sampling_steps: int = Field(10, ge=1)
    schedule_config: AnyDiffusionScheduleConfig = Field(
        default_factory=FlowMatchingConfig,
        discriminator="kind",
    )


class DataspaceDiffusionChunkedEvalCallback(AeroMetricsCallback):
    """Chunked full-mesh eval that samples via the configured schedule per chunk.

    All non-sampling behaviour — dataset_key, chunk_properties, chunk_size,
    forward_properties — matches the tutorial callback.
    """

    def __init__(self, callback_config, **kwargs):
        super().__init__(callback_config, **kwargs)
        self.sampling_steps = getattr(callback_config, "sampling_steps", 10)
        self.schedule_config = callback_config.schedule_config
        schedule = build_schedule(self.schedule_config)
        if not isinstance(schedule, FlowMatchingSchedule):
            raise ValueError(
                f"DataspaceDiffusionChunkedEvalCallback only supports FlowMatchingSchedule, got {type(schedule).__name__}"
            )
        self._schedule = schedule

    def _sample_chunk(self, chunked_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run FM Euler on one chunk — returns per-field tensors (1, chunk, dim_f).

        Uses ``schedule.sample_joint`` with a direct ``model(...)`` forward rather
        than ``model.sample()`` — the latter's ``_expand`` broadcast path assumes
        geometry tensors are batched on dim 0, but AB-UPT uses flat-concat
        geometry so ``geometry_position.shape[0]`` is ``N_geom_total`` (not batch).
        """
        model = self.model
        assert isinstance(model, DiffusionABUPT), (
            f"DataspaceDiffusionChunkedEvalCallback expects DiffusionABUPT, got {type(model).__name__}"
        )
        device = next(model.parameters()).device
        schedule = self._schedule.to(device)

        domain_names = list(model.domain_names)
        domain_anchor_positions: dict[str, torch.Tensor] = {
            name: chunked_batch[f"{name}_anchor_position"] for name in domain_names
        }
        ref_pos = domain_anchor_positions[domain_names[0]]
        B = ref_pos.shape[0]
        domain_shapes = [
            (B, domain_anchor_positions[name].shape[1], model.data_specs.domains[name].output_dims.total_dim)
            for name in domain_names
        ]

        geo_kw = {
            "geometry_position": chunked_batch["geometry_position"],
            "geometry_supernode_idx": chunked_batch["geometry_supernode_idx"],
            "geometry_batch_idx": chunked_batch["geometry_batch_idx"],
        }

        # Reuse the geometry encoding across Euler steps — same scene, only
        # the noisy fields and timestep change. First call runs the geometry
        # encoder; subsequent calls pass back ``{"geometry_encoding": tensor}``
        # so the backbone short-circuits the encoder while still re-running
        # anchors with the new features. We bypass ``DiffusionABUPT.forward``
        # to access the kv_cache — the wrapper drops it from its return.
        geometry_cache: dict[str, torch.Tensor] = {}

        def joint_fn(xt_list, t, _cond):
            domain_anchor_features = dict(zip(domain_names, xt_list, strict=True))
            backbone_kwargs: dict[str, Any] = dict(
                domain_anchor_positions=domain_anchor_positions,
                domain_anchor_features=domain_anchor_features,
                conditioning_inputs={"timestep": t.view(-1, 1)},
                kv_cache=geometry_cache,
            )
            if "geometry_encoding" not in geometry_cache:
                # First step: pass geometry inputs so the encoder runs and
                # populates ``geometry_encoding`` + ``geometry_rope`` in the
                # returned cache.
                backbone_kwargs.update(geo_kw)
            predictions, new_cache = model.backbone(**backbone_kwargs)
            if "geometry_encoding" not in geometry_cache:
                geometry_cache["geometry_encoding"] = new_cache["geometry_encoding"]
                geometry_cache["geometry_rope"] = new_cache["geometry_rope"]
            # Backbone returns per-field anchor predictions keyed ``{name}_{field}``;
            # concatenate them in field order into one (B, N, total_output_dims) noise
            # tensor per domain so the FM scheduler can integrate it as a single state.
            return [
                torch.cat(
                    [
                        predictions[f"{name}_{field}"]
                        for field in model.data_specs.domains[name].output_dims.field_slices
                    ],
                    dim=-1,
                )
                for name in domain_names
            ]

        # eval in fp32: disable autocast so Euler integration is deterministic
        # and per-field denormalized metrics aren't polluted by bf16 roundoff.
        with torch.autocast(device_type=device.type, enabled=False), torch.no_grad():
            x_list = schedule.sample_joint(
                domain_shapes,
                joint_fn,
                steps=self.sampling_steps,
            )

        result: dict[str, torch.Tensor] = {}
        for name, x in zip(domain_names, x_list, strict=True):
            for field, sl in model.data_specs.domains[name].output_dims.field_slices.items():
                result[f"{name}_{field}"] = x[..., sl]
        return result

    def _chunked_model_inference(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Chunked FM sampling. Mirrors the parent signature + concat logic."""
        batch_size = batch[self.sample_size_property].shape[1]
        chunk_indices = self._get_chunk_indices(batch_size)

        model_outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
        for start_idx, end_idx in chunk_indices:
            chunked_batch = self._create_chunked_batch(batch, start_idx, end_idx)
            chunk_out = self._sample_chunk(chunked_batch)
            for key, value in chunk_out.items():
                model_outputs[key].append(value)

        return {key: torch.cat(chunks, dim=1) for key, chunks in model_outputs.items()}

    def _run_model_inference(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Route non-chunked path through the schedule too (keeps field keys consistent)."""
        if self.chunked_inference:
            return self._chunked_model_inference(batch)
        return self._sample_chunk(batch)
