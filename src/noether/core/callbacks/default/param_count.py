#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.callbacks.base import CallbackBase
from noether.core.models import CompositeModel
from noether.core.utils.common import snake_type_name
from noether.core.utils.logging import short_number_str


class ParamCountCallback(CallbackBase):
    """Callback to log the number of trainable and frozen parameters of the model.

    This callback is initialized by the :class:`~noether.training.trainers.BaseTrainer` and should not be added
    manually to the trainer's callbacks.
    """

    @staticmethod
    def _get_param_counts(model, trace=None):
        if isinstance(model, CompositeModel):
            result = []
            immediate_children = []
            for name, submodel in model.submodels.items():
                if submodel is None:
                    continue
                subresult = ParamCountCallback._get_param_counts(submodel, trace=f"{trace}.{name}")
                result += subresult
                immediate_children.append(subresult[0])
            trainable_sum = sum(count for _, count, _ in immediate_children)
            frozen_sum = sum(count for _, _, count in immediate_children)
            return [(trace, trainable_sum, frozen_sum)] + result
        else:
            return [
                (
                    f"{snake_type_name(model)}" if trace is None else f"{trace}.{snake_type_name(model)}",
                    model.trainable_param_count,
                    model.frozen_param_count,
                )
            ]

    def before_training(self, **_) -> None:
        param_counts = self._get_param_counts(self.model)
        new_summary_entries = {}
        for name, tcount, fcount in param_counts:
            name = name or "total"
            self.logger.info(
                f"Parameter count of {name}: trainable={short_number_str(tcount)} frozen={short_number_str(fcount)}"
            )
            new_summary_entries[f"param_count/{name}/trainable"] = tcount
            new_summary_entries[f"param_count/{name}/frozen"] = fcount
            new_summary_entries[f"param_count/{name}"] = tcount + fcount
        self.tracker.update_summary(new_summary_entries)
