#  Copyright © 2025 Emmi AI GmbH. All rights reserved.


from noether.core.factory.base import Factory
from noether.core.schemas.schedules import AnyScheduleConfig
from noether.core.utils.training import ScheduleWrapper


class ScheduleFactory(Factory):
    """Factory for creating schedules. Handles wrapping into :class:`~noether.core.utils.training.schedule.ScheduleWrapper` which handles update/epoch based
    scheduling. Additionally, populates the ``effective_batch_size`` and ``updates_per_epoch`` to avoid specifying it
    in the config.
    """

    def create(self, schedule_config: AnyScheduleConfig, **kwargs) -> ScheduleWrapper | None:  # type: ignore[override]
        """Creates a schedule based on the provided config and wraps it into a :class:`~noether.core.utils.training.schedule.ScheduleWrapper`.

        Args:
            schedule_config: The schedule config or already instantiated schedule. See
                :class:`~noether.core.schemas.schedules.AnyScheduleConfig` for available options.
            **kwargs: Additional keyword arguments that are passed to the schedule constructor.

        Returns:
            The instantiated schedule wrapped in :class:`~noether.core.utils.training.schedule.ScheduleWrapper`.
        """
        if schedule_config is None:
            return None

        update_counter = kwargs.pop("update_counter", None)
        schedule = self.instantiate(schedule_config, **kwargs)

        return ScheduleWrapper(schedule=schedule, update_counter=update_counter, interval=schedule_config.interval)
