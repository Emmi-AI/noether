#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import math
from typing import Any

from noether.core.callbacks.base import CallBackBaseConfig
from noether.core.callbacks.periodic import PeriodicCallback
from noether.core.providers.metric_property import MetricPropertyProvider, Ordinality
from noether.core.schemas.lib import ConfiguredBy
from noether.core.utils.model import compute_model_norm


@ConfiguredBy(CallBackBaseConfig)
class TrainingDiagnosticsCallback(PeriodicCallback):
    """Logs gradient norms, the grad-scaler scale and the model norm, plus all losses after
    each accumulation step.

    Useful for monitoring training and diagnosing exploding or vanishing gradients
    (i.e. convergence speed / training dynamics).

    This callback is *not* added by default. Enable it via the ``callbacks`` config with::

        - kind: noether.core.callbacks.default.training_diagnostics.TrainingDiagnosticsCallback
          every_n_updates: 50

    using exactly one of ``every_n_updates`` / ``every_n_epochs`` / ``every_n_samples``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_logged_grad_scaler_scale: float | None = None
        self._register_metric_patterns()

    @staticmethod
    def _register_metric_patterns() -> None:
        """Register this callback's ``optim`` metric namespace as neutral.

        Without this, ``training_diagnostics/optim/{grad_scaler_scale,grad_norm/*,model_norm}``
        match no pattern in :class:`MetricPropertyProvider` (the default ``optim/*`` is anchored at
        the start of the key) and fall through to ``higher_is_better=True`` with a warning. The
        accumulation-step losses already match the default ``*loss*`` pattern, so only the ``optim``
        namespace needs registering here.
        """
        # Instantiating the provider triggers lazy registration of the default patterns if it has
        # not happened yet, so our pattern is appended *after* the defaults (FIFO match order means
        # defaults win on any overlap).
        MetricPropertyProvider()
        MetricPropertyProvider.register_pattern("training_diagnostics/optim/*", Ordinality.NEUTRAL)

    # noinspection PyMethodOverriding
    def periodic_callback(self, **_) -> None:
        grad_scaler = self.trainer.grad_scaler
        if not grad_scaler.is_enabled():
            return
        scale = grad_scaler.get_scale()
        if scale != self._last_logged_grad_scaler_scale:
            self.writer.add_scalar("training_diagnostics/optim/grad_scaler_scale", scale)
            self._last_logged_grad_scaler_scale = scale

        for cur_name, cur_model in self.model.get_named_models().items():
            optimizer = cur_model.optimizer
            if optimizer is None or optimizer.last_grad_norm is None:
                continue
            norm = optimizer.last_grad_norm.item()
            if math.isfinite(norm):
                self.writer.add_scalar(f"training_diagnostics/optim/grad_norm/{cur_name}", norm)
        model_norm = compute_model_norm(self.model).item()
        self.writer.add_scalar("training_diagnostics/optim/model_norm", model_norm)

    def track_after_accumulation_step(self, *, losses, **_) -> None:
        for loss, value in losses.items():
            self.writer.add_scalar(f"training_diagnostics/accumulation_step/loss/{loss}", value.item())
