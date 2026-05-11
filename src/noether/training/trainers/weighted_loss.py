#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import importlib
from collections.abc import Callable

import torch
import torch.nn.functional as F
from pydantic import Field

from noether.training.trainers import BaseTrainer, BaseTrainerConfig

LOSS_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {
    "mse": F.mse_loss,
    "l1": F.l1_loss,
    "smooth_l1": F.smooth_l1_loss,
    "huber": F.huber_loss,
}


class WeightedLossTrainerConfig(BaseTrainerConfig):
    """Config for a generic trainer that computes weighted loss per output field.

    ``field_weights`` maps output field names to their loss weights. Keys must match model output dict keys.
    Target keys in the batch are expected to follow the ``<field_name>_target`` convention.

    Built-in loss example::

        WeightedLossTrainerConfig(
            kind="noether.training.trainers.WeightedLossTrainer",
            field_weights={"surface_pressure": 1.0, "volume_velocity": 1.0},
            loss_fn="l1",
        )

    Custom loss function from a downstream project::

        WeightedLossTrainerConfig(
            kind="noether.training.trainers.WeightedLossTrainer",
            field_weights={"surface_pressure": 1.0},
            loss_fn="my_project.losses.weighted_huber",
        )
    """

    field_weights: dict[str, float] = Field(
        ...,
        description="Mapping from output field name to its loss weight.",
    )
    loss_fn: str = Field(
        "mse",
        description="Loss function: a built-in name ('mse', 'l1', 'smooth_l1', 'huber') "
        "or a dotted import path to a custom callable with signature (input, target) -> Tensor.",
    )


def _resolve_loss_fn(loss_fn: str) -> Callable[..., torch.Tensor]:
    """Resolve a loss function by short name or dotted import path.

    Built-in short names: ``mse``, ``l1``, ``smooth_l1``, ``huber``.

    For custom loss functions, use a fully qualified dotted path::

        _resolve_loss_fn("my_project.losses.wing_pressure_loss")
    """
    if loss_fn in LOSS_REGISTRY:
        return LOSS_REGISTRY[loss_fn]

    # Treat as a dotted import path: "package.module.function_name":
    module_path, _, attr_name = loss_fn.rpartition(".")
    if not module_path or not attr_name:
        raise ValueError(
            f"Unknown loss_fn '{loss_fn}'. Use a built-in name ({', '.join(LOSS_REGISTRY)}) or a dotted import path "
            "(e.g., 'my_project.losses.my_loss_fn')."
        )
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ValueError(f"Could not import module '{module_path}' for loss_fn '{loss_fn}'.") from exc
    try:
        fn = getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(f"Module '{module_path}' has no attribute '{attr_name}'.") from exc
    if not callable(fn):
        raise ValueError(f"'{loss_fn}' resolved to {type(fn).__name__}, expected a callable.")
    return fn  # type: ignore[no-any-return]


class WeightedLossTrainer(BaseTrainer):
    """Generic trainer that computes weighted loss per output field.

    Expects the model forward to return ``dict[str, Tensor]`` with keys matching
    ``field_weights`` keys, and the batch to contain ``<field_name>_target`` keys.

    The loss function defaults to MSE and can be changed via the ``loss_fn`` config
    parameter. Use a built-in short name or a dotted import path for custom losses.

    Built-in losses::

        trainer_params = dict(field_weights={"pressure": 1.0}, loss_fn="l1")

    Custom loss function from your project::

        trainer_params = dict(
            field_weights={"pressure": 1.0},
            loss_fn="my_project.losses.weighted_huber",
        )

    The custom callable must have the signature ``(input, target) -> Tensor``,
    matching ``torch.nn.functional`` loss functions.
    """

    def __init__(self, trainer_config: WeightedLossTrainerConfig, **kwargs):
        super().__init__(config=trainer_config, **kwargs)

        self._loss_fn = _resolve_loss_fn(trainer_config.loss_fn)

        self.loss_items: list[tuple[str, float]] = []
        for target_prop in self.target_properties:
            field_name = target_prop.removesuffix("_target")
            weight = trainer_config.field_weights.get(field_name)
            if weight is None:
                raise ValueError(
                    f"Target property '{target_prop}' (field '{field_name}') "
                    f"not found in field_weights. Available: {list(trainer_config.field_weights.keys())}"
                )
            self.loss_items.append((field_name, weight))

    def loss_compute(
        self,
        forward_output: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for field_name, weight in self.loss_items:
            if weight <= 0 or field_name not in forward_output:
                continue
            target_key = f"{field_name}_target"
            if target_key not in targets:
                raise ValueError(f"Target '{target_key}' not found in targets. Available: {list(targets.keys())}")
            losses[f"{field_name}_loss"] = self._loss_fn(targets[target_key], forward_output[field_name]) * weight
        if not losses:
            raise ValueError(
                "No losses computed. Check that 'field_weights' keys match model output keys and 'target_properties'."
            )
        return losses
