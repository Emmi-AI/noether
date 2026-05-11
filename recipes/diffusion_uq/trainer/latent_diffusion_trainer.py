#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch
from pydantic import Field
from torch import Tensor

from noether.core.schemas.callbacks import CallBackBaseConfig
from noether.core.schemas.diffusion import AnyDiffusionScheduleConfig, FlowMatchingConfig
from noether.core.schemas.trainers import BaseTrainerConfig
from noether.modeling.diffusion import build_schedule
from noether.training.trainers import BaseTrainer
from noether.training.trainers.types import TrainerResult


class LatentDiffusionTrainerConfig(BaseTrainerConfig[CallBackBaseConfig]):
    """Trainer config for diffusion stages.

    The schedule is a discriminated-union slot
    (:data:`~noether.core.schemas.diffusion.AnyDiffusionScheduleConfig`); pass
    a concrete :class:`~noether.core.schemas.diffusion.FlowMatchingConfig`.
    """

    schedule_config: AnyDiffusionScheduleConfig = Field(
        default_factory=FlowMatchingConfig,
        discriminator="kind",
    )
    """Diffusion / flow-matching schedule. Pydantic resolves the variant by ``kind``."""

    latent_scale: float | None = None
    """Optional scalar latent scaling. Ignored when ``latent_stats_path`` is set."""

    latent_stats_path: str | None = None
    """Path to ``train_stats.pt`` for per-token latent normalization
    (overrides :attr:`latent_scale`)."""

    field_loss_weights: dict[str, float] = Field(default_factory=dict)
    """Per-field MSE loss weights, keyed by ``{domain}_{field}`` (e.g.
    ``"surface_pressure"`` / ``"volume_velocity"``). Fields without an entry
    default to weight ``1.0``. Used by ``DiffusionABUPTTrainer`` to combine
    per-field losses into the total."""


class LatentDiffusionTrainer(BaseTrainer):
    """Denoiser trainer in latent space with geometry conditioning.

    Passes ``supernode_positions`` as per-token geometry conditioning. Latent
    normalization modes (mutually exclusive, checked in order):
        1. ``latent_stats_path`` → per-token z-score ``(x - mean) / std``.
        2. ``latent_scale`` → scalar scaling ``x * scale``.
        3. neither → identity.
    """

    def __init__(self, trainer_config: LatentDiffusionTrainerConfig, **kwargs):
        super().__init__(config=trainer_config, **kwargs)

        self.schedule = build_schedule(trainer_config.schedule_config)

        self.latent_scale = trainer_config.latent_scale or 1.0
        self._schedule_on_device = False

        # per-token stats override scalar latent_scale when present.
        self._latent_mean: Tensor | None = None
        self._latent_std: Tensor | None = None
        if trainer_config.latent_stats_path:
            stats = torch.load(trainer_config.latent_stats_path, weights_only=True)
            self._latent_mean = stats["latent_mean"]
            self._latent_std = stats["latent_std"]

    @property
    def per_token_norm(self) -> bool:
        return self._latent_mean is not None

    def _normalize(self, latents: Tensor) -> Tensor:
        if self._latent_mean is not None:
            mean = self._latent_mean.to(latents.device)
            std = self._latent_std.to(latents.device).clamp(min=1e-6)
            return (latents - mean) / std
        return latents * self.latent_scale

    def train_step(self, batch: dict[str, Tensor], model: torch.nn.Module) -> TrainerResult:
        latents = batch["latents"]
        supernode_positions = batch.get("supernode_positions")

        if not self._schedule_on_device:
            self.schedule.to(latents.device)
            self._schedule_on_device = True

        latents = self._normalize(latents)

        def model_fn(noisy_input: Tensor, timestep: Tensor, condition: Tensor | None) -> Tensor:
            return model(noisy_input, timestep=timestep, supernode_positions=supernode_positions)

        loss = self.schedule.training_losses(latents, model_fn, condition=None)

        return TrainerResult(total_loss=loss, losses_to_log={"diffusion_loss": loss})
