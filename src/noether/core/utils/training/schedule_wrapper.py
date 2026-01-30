#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from noether.core.schedules import ConstantSchedule, ScheduleBase
from noether.core.utils.training.counter import UpdateCounter


class ScheduleWrapper:
    """Wrapper around a schedule that handles getting the value based on an UpdateCounter and whether the schedule is
    based on updates or epochs."""

    def __init__(
        self,
        schedule: ScheduleBase,
        update_counter: UpdateCounter | None = None,
        interval: Literal["update", "epoch"] = "update",
    ):
        """
        Args:
            schedule: The :class:`noether.core.schedules.base.ScheduleBase` to wrap.
            update_counter: The :class:`noether.core.utils.training.counter.UpdateCounter` to use for getting the current step. If None, the schedule is assumed to
                be constant.
            interval: Whether the schedule is based on updates or epochs. Interval should be either "update" or "epoch".
        """

        self.schedule = schedule
        self.update_counter = update_counter
        if interval not in ["update", "epoch"]:
            raise NotImplementedError(f"invalid interval: {interval}")
        self.interval = interval

    def get_value(self) -> float:
        """Get the current value of the schedule based on the current step in the UpdateCounter."""
        if self.update_counter is None:
            if not isinstance(self.schedule, ConstantSchedule):
                raise ValueError("schedules other than ConstantSchedule require an UpdateCounter")
            return self.schedule.get_value(step=0, total_steps=1)

        if self.interval == "update":
            if self.update_counter.cur_iteration.update is None or self.update_counter.end_iteration.update is None:
                raise ValueError("UpdateCounter must have update checkpoints defined for 'update' interval")
            return self.schedule.get_value(
                step=self.update_counter.cur_iteration.update,
                total_steps=self.update_counter.end_iteration.update,
            )
        elif self.interval == "epoch":
            if self.update_counter.cur_iteration.epoch is None or self.update_counter.end_iteration.epoch is None:
                raise ValueError("UpdateCounter must have epoch checkpoints defined for 'epoch' interval")
            return self.schedule.get_value(
                step=self.update_counter.cur_iteration.epoch * self.update_counter.updates_per_epoch,
                total_steps=self.update_counter.end_iteration.epoch * self.update_counter.updates_per_epoch,
            )
        else:
            raise NotImplementedError(f"invalid interval: {self.interval}")
