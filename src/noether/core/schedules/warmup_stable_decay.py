#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schedules.base import ScheduleBase
from noether.core.schedules.constant import ConstantSchedule
from noether.core.schedules.cosine import CosineDecreasingSchedule
from noether.core.schedules.linear import LinearDecreasingSchedule, LinearIncreasingSchedule
from noether.core.schedules.sqrt import SqrtDecreasingSchedule
from noether.core.schemas.schedules import (
    ConstantScheduleConfig,
    DecreasingProgressScheduleConfig,
    IncreasingProgressScheduleConfig,
    WarmupStableDecayScheduleConfig,
)


class WarmupStableDecaySchedule(ScheduleBase):
    """Warmup-Stable-Decay schedule (Hägele et al. 2024).

    Three phases:
      1. Linear warmup ``start_value -> max_value`` over ``warmup_percent`` (or ``warmup_steps``).
      2. Constant ``max_value`` plateau.
      3. Decay ``max_value -> end_value`` over the final ``cooldown_percent`` (or ``cooldown_steps``).
         Cooldown shape selectable: ``"sqrt"`` (default, Hägele SOTA), ``"linear"``, or ``"cosine"``.

    Phase boundaries are dispatched manually inside ``_get_value`` because the cooldown is
    end-anchored (begins ``total_steps - cooldown_steps`` from the end), which the
    ``SequentialStepSchedule`` machinery cannot express at construction time.

    Example:

        .. code-block:: yaml

            schedule_config:
                kind: noether.core.schedules.WarmupStableDecaySchedule
                warmup_percent: 0.02
                cooldown_percent: 0.10
                cooldown_shape: sqrt
                end_value: 0.0
                max_value: ${model.optim.lr}
    """

    _COOLDOWN_CLASS_BY_SHAPE = {
        "sqrt": SqrtDecreasingSchedule,
        "linear": LinearDecreasingSchedule,
        "cosine": CosineDecreasingSchedule,
    }

    def __init__(self, config: WarmupStableDecayScheduleConfig):
        super().__init__(overhang_percent=config.overhang_percent, overhang_steps=config.overhang_steps)
        self.warmup_steps = config.warmup_steps
        self.warmup_percent = config.warmup_percent
        self.cooldown_steps = config.cooldown_steps
        self.cooldown_percent = config.cooldown_percent
        self.cooldown_shape = config.cooldown_shape
        self.end_value = config.end_value
        self.max_value = config.max_value

        self._warmup = LinearIncreasingSchedule(
            config=IncreasingProgressScheduleConfig.model_validate(
                dict(
                    exclude_first=config.start_value == 0,
                    exclude_last=True,
                    start_value=config.start_value,
                    max_value=config.max_value,
                )
            )
        )
        self._stable = ConstantSchedule(config=ConstantScheduleConfig.model_validate(dict(value=config.max_value)))
        cooldown_cls = self._COOLDOWN_CLASS_BY_SHAPE[config.cooldown_shape]
        self._cooldown = cooldown_cls(
            config=DecreasingProgressScheduleConfig.model_validate(
                dict(
                    exclude_first=False,
                    exclude_last=False,
                    max_value=config.max_value,
                    end_value=config.end_value,
                )
            )
        )

    def _phase_boundaries(self, total_steps: int) -> tuple[int, int]:
        """Return ``(warmup_end, cooldown_start)`` in absolute steps."""
        if self.warmup_steps is not None:
            assert self.cooldown_steps is not None
            return self.warmup_steps, total_steps - self.cooldown_steps
        # Match SequentialPercentSchedule's int(...) truncation so boundary semantics
        # stay consistent with LinearWarmupCosineDecaySchedule.
        assert self.warmup_percent is not None and self.cooldown_percent is not None
        total_steps_m1 = total_steps - 1
        warmup_end = int(self.warmup_percent * total_steps_m1)
        cooldown_start = int((1.0 - self.cooldown_percent) * total_steps_m1)
        return warmup_end, cooldown_start

    def _get_value(self, step: int, total_steps: int) -> float:
        warmup_end, cooldown_start = self._phase_boundaries(total_steps)

        if step < warmup_end:
            return self._warmup.get_value(step=step, total_steps=warmup_end)
        if step < cooldown_start:
            # ConstantSchedule ignores its arguments
            return self._stable.get_value(step=0, total_steps=1)
        cooldown_total = total_steps - cooldown_start
        if cooldown_total <= 0:
            return self.end_value
        adj_step = min(step - cooldown_start, cooldown_total - 1)
        return self._cooldown.get_value(step=adj_step, total_steps=cooldown_total)

    def __str__(self):
        if self.warmup_percent is not None:
            return (
                f"{type(self).__name__}(warmup_percent={self.warmup_percent}, "
                f"cooldown_percent={self.cooldown_percent}, cooldown_shape={self.cooldown_shape})"
            )
        return (
            f"{type(self).__name__}(warmup_steps={self.warmup_steps}, "
            f"cooldown_steps={self.cooldown_steps}, cooldown_shape={self.cooldown_shape})"
        )
