#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

from noether.core.callbacks.periodic import PeriodicCallback
from noether.core.models import CompositeModel
from noether.core.optimizer.muon_composite import MuonComposite

if TYPE_CHECKING:
    from noether.core.utils.training.training_iteration import TrainingIteration


class MuonAlphaCallback(PeriodicCallback):
    """Logs Muon α and two cheap per-step diagnostics for models using
    :class:`~noether.core.optimizer.MuonComposite`:

    * ``optim/muon/alpha/{model}``: current α value.
    * ``optim/muon/update_frob_mean/{model}``: mean Frobenius norm of the blended update
      across Muon params. Sanity-check that the Frobenius-matching ``√min(m, n)`` factor
      keeps the update scale ~constant across α.
    * ``optim/muon/cos_mhat_ns/{model}``: mean cosine similarity between ``M_hat`` and
      ``NS(M_hat)``. Tells us whether NS is redirecting the update vs. just reshaping its
      spectrum — drops indicate NS is doing real directional work.

    This callback is initialized by the :class:`~noether.training.trainers.BaseTrainer`
    and should not be added manually.
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
            torch_optim = cur_model.optimizer.torch_optim
            if not isinstance(torch_optim, MuonComposite) or torch_optim._muon is None:
                continue
            self.writer.add_scalar(f"optim/muon/alpha/{cur_name}", torch_optim.last_alpha)
            self.writer.add_scalar(f"optim/muon/update_frob_mean/{cur_name}", torch_optim.last_update_frob_mean)
            self.writer.add_scalar(f"optim/muon/cos_mhat_ns/{cur_name}", torch_optim.last_cos_mhat_ns)
