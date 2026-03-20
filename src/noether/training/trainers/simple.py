#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch

from noether.core.factory import class_constructor_from_class_path
from noether.core.schemas import BaseTrainerConfig
from noether.training.trainers.base import BaseTrainer
from noether.training.trainers.types import LossResult


class SimpleLossTrainerConfig(BaseTrainerConfig):
    """Config for :class:`SimpleLossTrainer`.

    Set ``forward_properties`` and ``target_properties`` to control which batch
    keys are forwarded to the model and which are treated as targets.

    Example YAML:

    .. code-block:: yaml

        kind: noether.training.trainers.SimpleLossTrainer
        loss_fn: torch.nn.functional.cross_entropy
        forward_properties: [x]
        target_properties: [y]
        max_epochs: 50
        effective_batch_size: 32
    """

    loss_fn: str
    """Dotted import path to a loss function, e.g. ``torch.nn.functional.cross_entropy``.

    The function is called as ``loss_fn(model_output, target)`` where ``target`` is the
    tensor from the first key in ``target_properties``.
    """


class SimpleLossTrainer(BaseTrainer):
    """Trainer for the common single-forward-pass → scalar-loss pattern.

    Eliminates the need for a custom trainer subclass in the majority of cases.
    Configure which batch keys flow to the model (``forward_properties``) and
    which are used as ground truth (``target_properties``), then specify the
    loss function by its dotted import path.

    The loss function is called as:

    .. code-block:: python

        loss = loss_fn(model_output, target)

    where ``model_output`` is the return value of ``model(**forward_properties_batch)``
    and ``target`` is the tensor for the first key in ``target_properties``.

    Example YAML:

    .. code-block:: yaml

        kind: noether.training.trainers.SimpleLossTrainer
        loss_fn: torch.nn.functional.cross_entropy
        forward_properties: [x]
        target_properties: [y]
    """

    def __init__(self, config: SimpleLossTrainerConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._loss_fn = class_constructor_from_class_path(config.loss_fn)

    def loss_compute(
        self,
        forward_output: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> LossResult:
        target = next(iter(targets.values()))
        return self._loss_fn(forward_output, target)
