#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

from noether.core.callbacks.periodic import PeriodicCallback
from noether.core.models import CompositeModel

if TYPE_CHECKING:
    from noether.core.utils.training.training_iteration import TrainingIteration


class LrCallback(PeriodicCallback):
    """Callback to log the learning rate of the optimizer.

    This callback is initialized by the :class:`~noether.training.trainers.BaseTrainer` and should not be added
    manually to the trainer's callbacks.
    """

    def _should_invoke_after_update(self, training_iteration: TrainingIteration):
        if training_iteration.update == 1:
            return True
        return super()._should_invoke_after_update(training_iteration)

    # noinspection PyMethodOverriding
    def periodic_callback(self, **_) -> None:
        for cur_name, cur_model in self.model.get_named_models().items():
            if isinstance(cur_model, CompositeModel) or cur_model.optimizer is None:
                continue
            for param_group in cur_model.optimizer.torch_optim.param_groups:
                group_name = f"/{param_group['name']}" if "name" in param_group else ""
                if cur_model.optimizer.schedule is not None:
                    lr = param_group["lr"]
                    self.writer.add_scalar(f"optim/lr/{cur_name}{group_name}", lr)
                if cur_model.optimizer.weight_decay_schedule is not None:
                    wd = param_group["weight_decay"]
                    self.writer.add_scalar(f"optim/wd/{cur_name}{group_name}", wd)
