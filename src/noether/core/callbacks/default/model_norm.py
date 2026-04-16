#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import torch

from noether.core.callbacks.periodic import PeriodicCallback
from noether.core.schemas.callbacks import ModelNormCallbackConfig
from noether.core.utils.model import compute_model_norm


class ModelNormCallback(PeriodicCallback):
    def __init__(self, callback_config: ModelNormCallbackConfig, **kwargs):
        super().__init__(callback_config, **kwargs)
        self.callback_config = callback_config
        self._model_norms: dict[int, dict[str, float]] = {}

    @torch.no_grad()
    def _compute_model_norm(self) -> dict[str, float]:
        update = self.trainer.update_counter.update
        norms_dict: dict[str, float] = {
            "model/full_norm": compute_model_norm(self.model).item(),
        }
        if self.callback_config.individual_params_norms:
            norms_dict.update(
                {f"model/param_norm/{name}": p.norm().item() for name, p in self.model.named_parameters()}
            )

        self._model_norms[update] = norms_dict

        evict = update - self.callback_config.history_steps
        if evict in self._model_norms:
            # remove old norms to keep the history buffer bounded
            del self._model_norms[evict]
        return norms_dict

    # noinspection PyMethodOverriding
    def periodic_callback(self, *, interval_type, **_) -> None:
        update = self.trainer.update_counter.update
        norms_dict = self._model_norms.get(update) or self._compute_model_norm()
        for name, norm in norms_dict.items():
            self.writer.add_scalar(name, norm)

    def track_after_update_step(self, **_) -> None:
        self._compute_model_norm()

    # def after_training(self, *, update_counter) -> None:
    #    if not update_counter.is_finished:
    #        # If training is not finished, it is likely due to an error; log the model norms history for debugging
    #        update = update_counter.update
    #        if update not in self._model_norms:
    #            self._compute_model_norm()
    #        for iter_update, norms_dict in self._model_norms.items():
    #            for norm_name, norm_value in norms_dict.items():
    #                self.writer.add_scalar(f"{norm_name}/history/i_{iter_update}", norm_value)
