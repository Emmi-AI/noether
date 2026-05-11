#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch
import torch.nn.functional as F
from models.diffusion_abupt import DiffusionABUPT
from pydantic import Field
from torch import Tensor

from noether.core.schemas.callbacks import CallBackBaseConfig
from noether.core.schemas.trainers import BaseTrainerConfig
from noether.modeling.diffusion import (
    AnyDiffusionScheduleConfig,
    FlowMatchingConfig,
    FlowMatchingSchedule,
    build_schedule,
)
from noether.training.trainers import BaseTrainer
from noether.training.trainers.types import TrainerResult


class DiffusionTrainerConfig(BaseTrainerConfig[CallBackBaseConfig]):
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

    field_loss_weights: dict[str, float] = Field(default_factory=dict)
    """Per-field MSE loss weights, keyed by ``{domain}_{field}`` (e.g.
    ``"surface_pressure"`` / ``"volume_velocity"``). Fields without an entry
    default to weight ``1.0``. Used by ``DiffusionABUPTTrainer`` to combine
    per-field losses into the total."""


class DiffusionABUPTTrainer(BaseTrainer):
    """Joint flow-matching trainer for AB-UPT dataspace diffusion.

    Builds the per-domain ``domain_anchor_positions`` / ``domain_anchor_features``
    dicts directly from the batch and forwards them to
    :meth:`DiffusionABUPT.forward` under the same names — no per-domain unrolling.
    The total loss is a weighted sum of per-field MSE losses; channel ranges
    come from ``model.data_specs.domains[name].output_dims.field_slices``, and
    weights from :attr:`DiffusionTrainerConfig.field_loss_weights`
    (default ``1.0`` per field).
    """

    def __init__(self, trainer_config: DiffusionTrainerConfig, **kwargs):
        super().__init__(config=trainer_config, **kwargs)
        self.field_loss_weights = dict(trainer_config.field_loss_weights)
        schedule = build_schedule(trainer_config.schedule_config)

        if not isinstance(schedule, FlowMatchingSchedule):
            raise NotImplementedError("DiffusionABUPTTrainer currently supports FlowMatchingSchedule only.")
        self.schedule = schedule

        self._schedule_on_device = False

    def train_step(self, batch: dict[str, Tensor], model: torch.nn.Module) -> TrainerResult:
        assert isinstance(model, DiffusionABUPT), (
            f"DiffusionABUPTTrainer expects DiffusionABUPT, got {type(model).__name__}"
        )

        if not self._schedule_on_device:
            device = next(v for v in batch.values() if isinstance(v, Tensor)).device
            self.schedule.to(device)
            self._schedule_on_device = True

        domain_anchor_positions: dict[str, Tensor] = {}
        domain_targets: dict[str, Tensor] = {}
        for name, spec in model.data_specs.domains.items():
            domain_anchor_positions[name] = batch[f"{name}_anchor_position"]
            domain_targets[name] = torch.cat(
                [batch[f"{name}_{field}_target"] for field in spec.output_dims.field_slices],
                dim=-1,
            )

        # shared t across all branches for joint FM noising.
        ref = next(iter(domain_targets.values()))
        t = self.schedule._sample_time(ref.shape[0], ref.device)

        domain_anchor_features: dict[str, Tensor] = {}
        domain_velocity_targets: dict[str, Tensor] = {}
        for name, fields in domain_targets.items():
            xt, v_tgt = self.schedule.noise_pair(fields, t)
            domain_anchor_features[name] = xt
            domain_velocity_targets[name] = v_tgt

        out = model(
            timestep=t,
            geometry_position=batch["geometry_position"],
            geometry_supernode_idx=batch["geometry_supernode_idx"],
            geometry_batch_idx=batch["geometry_batch_idx"],
            domain_anchor_positions=domain_anchor_positions,
            domain_anchor_features=domain_anchor_features,
        )

        total_loss = torch.zeros((), device=ref.device)
        losses_to_log: dict[str, Tensor] = {}
        for name, target in domain_velocity_targets.items():
            for field, slc in model.data_specs.domains[name].output_dims.field_slices.items():
                key = f"{name}_{field}"
                weight = self.field_loss_weights.get(key, 1.0)
                field_loss = F.mse_loss(out[key], target[..., slc])
                total_loss = total_loss + weight * field_loss
                losses_to_log[f"{key}_v_loss"] = field_loss.detach()

        return TrainerResult(total_loss=total_loss, losses_to_log=losses_to_log)
