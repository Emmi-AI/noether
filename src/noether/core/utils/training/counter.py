#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.utils.training.training_iteration import TrainingIteration


class UpdateCounter:
    """UpdateCounter is used to track the current update, epoch, and sample
    number during training."""

    def __init__(
        self,
        start_iteration: TrainingIteration,
        end_iteration: TrainingIteration,
        updates_per_epoch: int,
        effective_batch_size: int,
    ):
        """Initializes UpdateCounter.

        Args:
            start_iteration: The iteration (measured in either epochs, updates, or samples) at which the training
                will start. If the training is resumed the values will be updated accordingly.
            end_iteration: The iteration (measured in either epochs, updates, or samples) at which the training will
                end.
            updates_per_epoch: Number of updates per epoch.
            effective_batch_size: Effective batch size.
        """
        self.updates_per_epoch = updates_per_epoch

        # start_checkpoint should always be fully specified (either E0_U0_S0 or derived from ResumeInitializer)
        self.start_iteration = start_iteration
        if not self.start_iteration.is_fully_specified:
            raise ValueError(f"start_iteration must be fully specified, got {self.start_iteration} instead")

        # fully specify end_checkpoint (based on difference between start_checkpoint)
        # resuming with a different batchsize is not supported
        # - batchsize can still be lower than the defined batchsize by only using a subset of the loaded batch
        # - schedules are not adjusted to it
        # - how are schedules such as inverse sqrt schedule handled?
        # use update instead of epoch here as epoch is int and therefore wouldn't work with update based ckpts
        assert self.start_iteration == TrainingIteration(update=self.start_iteration.update).to_fully_specified(
            updates_per_epoch=updates_per_epoch,
            effective_batch_size=effective_batch_size,
        )
        if not end_iteration.is_minimally_specified:
            raise ValueError(f"end_iteration must be minimally specified, got {end_iteration} instead")

        delta_iteration = end_iteration - self.start_iteration.to_target_specification(end_iteration)
        fully_specified_delta = delta_iteration.to_fully_specified(
            updates_per_epoch=updates_per_epoch,
            effective_batch_size=effective_batch_size,
        )

        self.end_iteration = self.start_iteration + fully_specified_delta
        # Recompute epoch from absolute updates to avoid truncation artifacts from delta-based computation
        # (e.g. mid-epoch resume: delta=679 updates truncates to 4 epochs, but absolute 750 updates = 5 epochs):
        if self.end_iteration.update is not None:
            self.end_iteration.epoch = self.end_iteration.update // updates_per_epoch
        if not self.end_iteration.is_fully_specified:
            raise ValueError(f"end_iteration must be fully specified, got {self.end_iteration} instead")

        self.cur_iteration = self.start_iteration.copy()
        self.effective_batch_size = effective_batch_size

    @property
    def is_full_epoch(self) -> bool:
        """Returns True if the current update is a full epoch."""
        assert self.cur_iteration.is_fully_specified and self.update is not None
        # The above assert already checks that cur_checkpoint.update is not None
        return self.update % self.updates_per_epoch == 0

    @property
    def epoch_as_float(self) -> float:
        """Returns the current epoch as a float value."""
        assert self.cur_iteration.update is not None
        return float(self.cur_iteration.update) / self.updates_per_epoch

    @property
    def epoch(self) -> int | None:
        """Returns the current epoch."""
        return self.cur_iteration.epoch

    @property
    def update(self) -> int | None:
        """Returns the current update."""
        return self.cur_iteration.update

    @property
    def sample(self) -> int | None:
        """Returns the current sample."""
        return self.cur_iteration.sample

    @property
    def is_finished(self) -> bool:
        """Returns True if the end checkpoint is reached."""
        return self.cur_iteration.to_target_specification(self.end_iteration) >= self.end_iteration

    def next_epoch(self) -> None:
        """Increments the current epoch by 1."""
        assert self.cur_iteration.epoch is not None
        self.cur_iteration.epoch += 1

    def next_update(self) -> None:
        """Increments the current update by 1."""
        assert self.cur_iteration.update is not None
        self.cur_iteration.update += 1

    def add_samples(self, num_samples: int) -> None:
        """Adds samples to the current sample count."""
        assert self.cur_iteration.sample is not None
        self.cur_iteration.sample += num_samples
