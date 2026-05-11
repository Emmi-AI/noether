#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Sample-and-decode eval callback for latent-space AB-UPT diffusion.

Mirrors the structure of :class:`DataspaceDiffusionChunkedEvalCallback`: a
periodic data-iterator callback that swaps the regression forward for a
schedule-driven sampling step. Here the model is a
:class:`LatentDiffusionModel` (frozen AB-UPT autoencoder + trainable latent
denoiser); after sampling latents from noise we run
``model.autoencoder.decode`` to project them back to per-field anchor
predictions, then compute MSE / MAE / relative-L2 errors against the
ground-truth fields delivered in the same batch.

Comparison happens in the AE's normalized field space — the latent dataset
wraps the original CFD dataset and forwards anchor positions, geometry,
super positions, and ground-truth ``{domain}_{field}_target`` keys through
the standard AB-UPT pipeline, so the batch already carries everything the
callback needs (no out-of-band disk reads). Metric keys follow the
``loss/<dataset_key>/<field>_<metric>`` convention used by the dataspace
eval, so latent and dataspace runs are directly comparable when they share
an AE.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import Field

from noether.core.callbacks.periodic import PeriodicDataIteratorCallback
from noether.core.schemas.callbacks import PeriodicDataIteratorCallbackConfig
from noether.core.schemas.diffusion import AnyDiffusionScheduleConfig, FlowMatchingConfig
from noether.modeling.diffusion import build_schedule
from noether.modeling.diffusion.flow_matching import FlowMatchingSchedule

METRIC_PREFIX_LOSS = "loss/"


class DiffusionEvalCallbackConfig(PeriodicDataIteratorCallbackConfig):
    """Config for sample-and-decode eval on latent-space diffusion."""

    name: Literal["DiffusionEvalCallback"] = "DiffusionEvalCallback"
    sampling_steps: int = Field(10, ge=1)
    schedule_config: AnyDiffusionScheduleConfig = Field(
        default_factory=FlowMatchingConfig,
        discriminator="kind",
    )
    """Schedule used for sampling at eval time. Independent of the trainer's
    schedule so callers can swap (e.g. EDM-trained model, FM eval)."""


class DiffusionEvalCallback(PeriodicDataIteratorCallback):
    """Periodic eval: sample latents conditioned on geometry, decode via AE, log per-field metrics.

    For each test sample:

    1. Sample latents with the configured schedule, conditioned on the
       sample's ``supernode_positions`` (per-token geometry conditioning,
       same as during denoiser training).
    2. Inverse the trainer's latent normalization so the AE sees latents in
       its training distribution.
    3. Run :meth:`ABUPTAutoencoder.decode` with batch-supplied anchor /
       super / geometry positions, yielding per-field anchor predictions.
    4. Compare predictions to the batch's ``{domain}_{field}_target``
       tensors and accumulate MSE / MAE / relative-L2 metrics.

    All inputs are read straight from the batch produced by
    :class:`LatentDiffusionPipeline` — the wrapped :class:`LatentDataset`
    re-runs the AB-UPT sample processors against the original CFD sample,
    so anchor positions, geometry tokens, and target fields are already
    present alongside the latent items.

    Restricted to batch_size=1: per-sample mesh tensors vary in length and
    would require padding; a single-sample iteration matches
    :class:`AeroMetricsCallback` and the dataspace eval.
    """

    def __init__(self, callback_config: DiffusionEvalCallbackConfig, **kwargs: Any):
        super().__init__(callback_config, **kwargs)
        self.sampling_steps = callback_config.sampling_steps
        self.schedule_config = callback_config.schedule_config
        schedule = build_schedule(self.schedule_config)
        if not isinstance(schedule, FlowMatchingSchedule):
            raise ValueError(f"DiffusionEvalCallback only supports FlowMatchingSchedule, got {type(schedule).__name__}")
        self._schedule = schedule

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Inverse of ``LatentDiffusionTrainer._normalize`` using trainer state.

        The denoiser is trained on normalized latents, so its sampler returns
        normalized latents — the AE expects raw encode-space latents.
        """
        trainer = self.trainer
        mean = getattr(trainer, "_latent_mean", None)
        std = getattr(trainer, "_latent_std", None)
        if mean is not None and std is not None:
            mean = mean.to(latents.device)
            std = std.to(latents.device).clamp(min=1e-6)
            return latents * std + mean
        scale = getattr(trainer, "latent_scale", 1.0) or 1.0
        return latents / scale

    @torch.no_grad()
    def _sample_latents(
        self,
        shape: tuple[int, ...],
        supernode_positions: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Run the schedule's Euler sampler with the denoiser as the velocity field."""
        denoiser = self.model.denoiser
        schedule = self._schedule.to(device)

        def model_fn(noisy: torch.Tensor, t: torch.Tensor, _cond: torch.Tensor | None) -> torch.Tensor:
            return denoiser(noisy, timestep=t, supernode_positions=supernode_positions)

        return schedule.sample(shape, model_fn, steps=self.sampling_steps)

    @torch.no_grad()
    def _decode(
        self,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Decode sampled latents to per-field anchor predictions via the frozen AE."""
        ae = self.model.autoencoder
        if ae is None:
            raise ValueError(
                "DiffusionEvalCallback requires the model to have a non-None `autoencoder` "
                "(e.g. a LatentDiffusionModel with autoencoder_config set)."
            )

        domain_anchor_positions: dict[str, torch.Tensor] = {}
        domain_super_positions: dict[str, torch.Tensor | None] = {}
        for name in ae.domain_names:
            anchor = batch.get(f"{name}_anchor_position")
            if anchor is not None:
                domain_anchor_positions[name] = anchor
            domain_super_positions[name] = batch.get(f"super_position_{name}")

        return ae.decode(
            latents=latents,
            domain_anchor_positions=domain_anchor_positions,
            geometry_position=batch.get("geometry_position"),
            geometry_supernode_idx=batch.get("geometry_supernode_idx"),
            domain_super_positions=domain_super_positions,
        )

    def _field_metrics(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        field_key: str,
    ) -> dict[str, torch.Tensor]:
        """MSE / MAE / relative-L2, matching :class:`AeroMetricsCallback._compute_metrics`."""
        delta = prediction - target
        metrics: dict[str, torch.Tensor] = {
            f"{field_key}_mse": (delta**2).mean(),
            f"{field_key}_mae": delta.abs().mean(),
        }
        target_norm = target.norm()
        if target_norm > 1e-8:
            metrics[f"{field_key}_l2err"] = delta.norm() / target_norm
        return metrics

    def process_data(self, batch: dict[str, torch.Tensor], **_: Any) -> dict[str, torch.Tensor]:
        """Sample latents → decode → per-field metrics for one (batch_size=1) sample."""
        if batch["latents"].shape[0] != 1:
            raise ValueError("DiffusionEvalCallback only supports batch_size=1")

        ae = self.model.autoencoder
        if ae is None:
            raise ValueError(
                "DiffusionEvalCallback requires the model to have a non-None `autoencoder` "
                "(e.g. a LatentDiffusionModel with autoencoder_config set)."
            )

        device = next(self.model.parameters()).device
        ref_latents = batch["latents"]
        supernode_positions = batch["supernode_positions"]

        with torch.autocast(device_type=device.type, enabled=False):
            sampled = self._sample_latents(ref_latents.shape, supernode_positions, device)
            ae_latents = self._denormalize_latents(sampled)
            preds = self._decode(ae_latents, batch)

        metrics: dict[str, torch.Tensor] = {}
        for domain in ae.domain_names:
            for field in ae.data_specs.domains[domain].output_dims.keys():
                field_key = f"{domain}_{field}"
                target = batch.get(f"{field_key}_target")
                pred = preds.get(field_key)
                if target is None or pred is None:
                    continue
                metrics.update(self._field_metrics(pred, target, field_key))

        # Also keep the latent-space MSE so latent-only progress (independent
        # of AE quality) stays observable.
        metrics["latent_mse"] = (sampled - ref_latents).pow(2).mean()
        return metrics

    def process_results(self, results: dict[str, torch.Tensor], **_: Any) -> None:
        """Log mean of every collated metric under ``loss/<dataset_key>/<key>``."""
        if not results:
            self.logger.warning(f"No metrics computed for dataset '{self.dataset_key}'")
            return
        for name, metric in results.items():
            self.writer.add_scalar(
                key=f"{METRIC_PREFIX_LOSS}{self.dataset_key}/{name}",
                value=metric.mean(),
                logger=self.logger,
                format_str=".6f",
            )
