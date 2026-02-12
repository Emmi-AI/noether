#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import dataclasses
import logging
from collections.abc import Callable, Iterator

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader, default_collate

from noether.core.utils.common import SizedIterable
from noether.data.samplers.internals import (
    SamplerOutput,
    _InterleavedBatchSampler,
    _InterleavedCollator,
    _InterleavedConcatDataset,
)

_logger = logging.getLogger(__name__)


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class SamplerIntervalConfig:
    """Configuration dataclass for setting up the dataloading pipeline, which is structured to load data from a "main"
    dataset (i.e., the dataset used for training), which is interleaved by iterations over other datasets (e.g., a
    test dataset to calculate a metric in a callback) in regular intervals.

    Args:
        sampler (SizedIterable): Any sampler that would be used in `torch.utils.data.DataLoader(sampler=...)`.
            Examples: `RandomSampler` for a training dataset or `SequentialSampler` for evaluation.
        every_n_epochs (int | None): Epoch-based interval. Invokes the callback after every n epochs. Mutually
            exclusive with other intervals.
        every_n_updates (int | None): Update-based interval. Invokes the callback after every n epochs. Mutually
            exclusive with other intervals.
        every_n_samples (int | None): Sample-based interval. Invokes the callback after every n epochs. Mutually
            exclusive with other intervals.
        pipeline (Optional[callable]): Any function that would be used in `torch.utils.data.DataLoader(collate_fn=...)`.
        batch_size (int | None): Batch size to use for this callback. Default: None (which will use the same batch_size
            as used for the "main" sampler, i.e., the one used for training).
    """

    sampler: SizedIterable
    pipeline: Callable | None
    every_n_epochs: int | None = None
    every_n_updates: int | None = None
    every_n_samples: int | None = None
    batch_size: int | None = None

    @model_validator(mode="after")
    def validate_frequency(self) -> "SamplerIntervalConfig":
        """
        Ensures that exactly one frequency ('every_n_*') is specified and
        that 'batch_size' is present if 'every_n_samples' is used.
        """
        frequency_fields = [self.every_n_epochs, self.every_n_updates, self.every_n_samples]
        num_frequency_fields_set = sum(1 for f in frequency_fields if f is not None)

        if num_frequency_fields_set != 1:
            raise ValueError(
                "Exactly one of 'every_n_epochs', 'every_n_updates', or 'every_n_samples' must be set. Cannot have multiple or none set."
            )

        if self.every_n_samples is not None and self.batch_size is None:
            raise ValueError("'batch_size' is required when 'every_n_samples' is set.")

        return self

    @field_validator("every_n_epochs", "every_n_updates", "every_n_samples", "batch_size", mode="after")
    @classmethod
    def check_positive_values(cls, v: int | None) -> int | None:
        """
        Ensures that all integer-based frequency and batch size fields are positive.
        """
        if v is not None and v <= 0:
            raise ValueError(f"Value must be a positive integer, but got {v}")
        return v


class InterleavedSamplerConfig(BaseModel):
    # properties of main sampler
    batch_size: int
    """batch_size to use for creating batches of the main_sampler indices."""

    drop_last: bool = True
    """Whether to drop the last non-full batch of the main_sampler."""

    # duration of InterleavedSampler
    max_epochs: int | None = None
    """How many epochs to sample at most from the main_sampler. Whatever limit is reached first (epochs/updates/samples) will stop the sampling."""
    max_updates: int | None = None
    """How many updates to sample at most from the main_sampler. Whatever limit is reached first (epochs/updates/samples) will stop the sampling."""
    max_samples: int | None = None
    """How many samples to sample at most from the main_sampler. Whatever limit is reached first (epochs/updates/samples) will stop the sampling."""
    start_epoch: int | None = None
    """At which epoch to start (used for resuming training). Mutually exclusive with `start_update` and `start_sample`."""
    start_update: int | None = None
    """At which update to start (used for resuming training). Mutually exclusive with `start_epoch` and `start_sample`."""
    start_sample: int | None = None
    """At which sample to start (used for resuming training). Mutually exclusive with `start_epoch` and `start_update`."""

    evaluation: bool = False
    """If True, the sampler is used for evaluation and will only iterate over the interleaved samplers once without iterating over the main sampler."""

    @field_validator(
        "start_epoch", "start_update", "start_sample", "batch_size", "max_epochs", "max_updates", "max_samples"
    )
    @classmethod
    def check_positive_values(cls, v: int | None) -> int | None:
        """
        Ensures that all integer-based frequency and batch size fields are positive.
        """
        if v is not None and v < 0:
            raise ValueError(f"Value must be a positive integer or zero, but got {v}")
        return v

    @model_validator(mode="after")
    def validate_stop(self) -> "InterleavedSamplerConfig":
        """
        Ensures that at least one frequency ('*_n_*') is specified and
        """
        stop_fields = [self.max_samples, self.max_updates, self.max_epochs]
        n_stop_fields = sum(1 for f in stop_fields if f is not None)

        if n_stop_fields == 0:
            raise ValueError("At least one of 'samples', 'updates', or 'epochs' must be set.")

        if self.max_samples is not None and self.batch_size is None:
            raise ValueError("'batch_size' is required when 'total_samples' is set.")

        return self

    @model_validator(mode="after")
    def validate_start(self) -> "InterleavedSamplerConfig":
        """
        Ensures that at least one start ('start_*') is specified
        """
        start_fields = [self.start_epoch, self.start_update, self.start_sample]
        n_start_fields = sum(1 for f in start_fields if f is not None)

        if n_start_fields > 1:
            raise ValueError("At most one of 'start_epoch', 'start_update', or 'start_sample' must be set.")

        if self.start_sample is not None:
            if self.batch_size is None:
                raise ValueError("'batch_size' is required when 'start_sample' is set.")
            elif self.start_sample % self.batch_size != 0:
                raise ValueError("'start_sample' must be a multiple of 'batch_size'.")

        return self


@dataclasses.dataclass
class _TrainingIterationState:
    epoch: int
    update: int
    sample: int
    sample_in_update: int = 0
    sample_at_last_update: int = 0
    sample_in_epoch: int = 0

    def next_sample(self, batch_size: int, samples_per_epoch: int):
        self.sample += 1
        self.sample_in_update += 1
        self.sample_in_epoch += 1

        return self.sample_in_update == batch_size or self.sample_in_epoch == samples_per_epoch

    def next_update(self, samples_per_epoch: int):
        self.sample_in_update = 0
        self.update += 1
        if self.sample_in_epoch == samples_per_epoch:
            _logger.debug(f"Completed epoch {self.epoch}")
            self.epoch += 1

    def start_new_epoch(self):
        _logger.debug(f"Starting epoch {self.epoch}")
        self.sample_in_epoch = 0

    def is_full_update(self, batch_size: int):
        return self.sample_in_update == batch_size

    def is_full_epoch(self, samples_per_epoch: int):
        return self.sample_in_epoch == samples_per_epoch


class InterleavedSampler:
    """Sampler to allow efficient dataloading by using a single large dataset containing train/test/... datasets all at
    once. The sampler will sample from different regionis in the dataset according to its specification. For example,
    consider a training dataset of length 100 and a test dataset of length 10. If the sampler is configured with a
    RandomSampler of the training dataset indices as main_sampler, it will repeatedly iterate over the training
    dataset. If the test dataset is configured with a sequential sampler that should be invoked after every epoch, the
    sampler will first return indices for the 100 training samples (randomly sampled) and then indices for the 10 test
    samples (in sequential order).

    Args:
        train_sampler: Sampler that is invoked by default (e.g., randomly sample from the trainset)
        config: Configuration for the InterleavedSampler.
        train_collator: Collator used to collate samples from indices sampled from the train sampler.
        callback_samplers: Configurations when the train_sampler should be paused and
            indices from other samplers (e.g., from a testset) should be returned. Also configures the interval and
            optionally a different batch_size to use for the interleaved batches.
    """

    def __init__(
        self,
        train_sampler: SizedIterable,
        config: InterleavedSamplerConfig,
        train_collator: Callable | None = None,
        callback_samplers: list[SamplerIntervalConfig] | None = None,
    ):
        super().__init__()

        callback_samplers = callback_samplers or []
        self.config = config
        self.main_sampler = train_sampler
        self.extra_samplers = callback_samplers
        self.start_epoch, self.start_update, self.start_sample = InterleavedSampler.calculate_start(
            config, len(train_sampler)
        )

        def _get_data_source(sampler: SizedIterable):
            if hasattr(sampler, "data_source"):
                return sampler.data_source
            if hasattr(sampler, "dataset"):
                return sampler.dataset
            raise NotImplementedError

        self.index_offsets = [len(_get_data_source(self.main_sampler))]
        for extra_sampler in self.extra_samplers[:-1]:
            self.index_offsets.append(self.index_offsets[-1] + len(_get_data_source(extra_sampler.sampler)))

        self.dataset = _InterleavedConcatDataset(
            [_get_data_source(self.main_sampler)]
            + [_get_data_source(extra_sampler.sampler) for extra_sampler in self.extra_samplers]
        )
        self.collator = _InterleavedCollator(
            [train_collator or default_collate]
            + [extra_sampler.pipeline or default_collate for extra_sampler in self.extra_samplers]
        )

        self.batch_sampler = _InterleavedBatchSampler(self)
        self.batch_size = config.batch_size

        if self.config.drop_last and len(self.main_sampler) > 0:
            if len(self.main_sampler) < self.batch_size:
                self.batch_size = len(self.main_sampler)
            batch_size = self.batch_size
            self.samples_per_epoch = len(self.main_sampler) // batch_size * batch_size
        else:
            self.samples_per_epoch = len(self.main_sampler)

    @staticmethod
    def calculate_start(config: InterleavedSamplerConfig, sampler_len: int):
        updates_per_epoch = sampler_len // config.batch_size

        if config.start_epoch is not None:
            start_update = updates_per_epoch * config.start_epoch
            start_sample = start_update * config.batch_size
            start_epoch = config.start_epoch
        elif config.start_update is not None:
            start_update = config.start_update
            start_epoch = int(start_update / updates_per_epoch)
            start_sample = start_update * config.batch_size
            if start_update % updates_per_epoch != 0 or not config.drop_last:
                raise NotImplementedError("defining start_update would require to skip forward in the sampler")
        elif config.start_sample is not None:
            start_sample = config.start_sample
            start_update = start_sample // config.batch_size
            start_epoch = int(start_update / updates_per_epoch)
            if start_update % updates_per_epoch != 0 or not config.drop_last:
                raise NotImplementedError("defining start_update would require to skip forward in the sampler")
        else:
            start_epoch = start_update = start_sample = 0

        return start_epoch, start_update, start_sample

    def get_data_loader(self, num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
        """Creates the DataLoader that uses the InterleavedSampler with the accordingly configured dataset.

        Args:
            num_workers: Number of workers to use.
            pin_memory: Whether to use pin memory.

        Returns:
            DataLoader that uses the InterleavedSampler with the accordingly configured dataset.
        """
        return DataLoader(
            dataset=self.dataset,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def __iter__(self) -> Iterator[SamplerOutput]:
        """Returns tuples of
        - bool: whether or not the sample is the last one in the batch
        - int: index of the sample that should be loaded

        Training loop will use the interleaved setup (sample from main_sampler by default and interleave configured
        samplers in between in regular intervals. Eval loop will not iterate over the main_sampler but only once over
        all interleaved samplers.

        Returns:
            Iterator for loading and batching samples.
        """
        if (
            self.config.max_epochs == 0
            or self.config.max_updates == 0
            or self.config.max_samples == 0
            or self.config.evaluation
        ):
            yield from self._eval_loop()
        else:
            yield from self._training_loop()

    def _eval_loop(self) -> Iterator[SamplerOutput]:
        for config_idx, config in enumerate(self.extra_samplers):
            yield from self._iterate_from_sampler(config, self.index_offsets[config_idx])

    def _end_reached(self, state: _TrainingIterationState):
        return (
            (self.config.max_epochs is not None and state.epoch >= self.config.max_epochs)
            or (self.config.max_updates is not None and state.update >= self.config.max_updates)
            or (self.config.max_samples is not None and state.sample >= self.config.max_samples)
        )

    def _sampler_needs_to_iterate(self, state: _TrainingIterationState, interval_config: SamplerIntervalConfig) -> bool:
        # can only occour at the end of an epoch
        if interval_config.every_n_epochs is not None and (
            state.sample_in_epoch == self.samples_per_epoch and state.epoch % interval_config.every_n_epochs == 0
        ):
            _logger.debug(f"Sampler {interval_config} needs to iterate at epoch {state.epoch}")
            return True

        if interval_config.every_n_updates is not None and state.update % interval_config.every_n_updates == 0:
            _logger.debug(f"Sampler {interval_config} needs to iterate at update {state.update}")
            return True

        if interval_config.every_n_samples is not None:
            if state.sample % interval_config.every_n_samples == 0:
                _logger.debug(f"Sampler {interval_config} needs to iterate at sample {state.sample}")
                return True

            if (
                state.sample_at_last_update // interval_config.every_n_samples
                < state.sample // interval_config.every_n_samples
            ):
                _logger.debug(f"Sampler {interval_config} needs to iterate at sample {state.sample}")
                return True

        return False

    def _iterate_from_sampler(self, config: SamplerIntervalConfig, index_offset: int) -> Iterator[SamplerOutput]:
        interleaved_batch_size = config.batch_size or self.batch_size
        sample_in_interleaved = 0
        _logger.debug(f"Iterating from sampler {config} with batch size {interleaved_batch_size}, {index_offset=}")
        for interleaved_idx in config.sampler:
            sample_in_interleaved += 1
            last_sample_in_update = sample_in_interleaved % interleaved_batch_size == 0 or sample_in_interleaved == len(
                config.sampler
            )

            yield SamplerOutput(is_full_batch=last_sample_in_update, idx=index_offset + interleaved_idx)

    def _training_loop(self) -> Iterator[SamplerOutput]:
        state = _TrainingIterationState(epoch=self.start_epoch, update=self.start_update, sample=self.start_sample)

        while True:
            if hasattr(self.main_sampler, "set_epoch") and callable(self.main_sampler.set_epoch):  # pyright: ignore[reportAttributeAccessIssue]
                self.main_sampler.set_epoch(state.epoch)  # pyright: ignore[reportAttributeAccessIssue]

            state.start_new_epoch()
            for main_idx in self.main_sampler:
                last_sample_in_update = state.next_sample(self.batch_size, self.samples_per_epoch)
                yield SamplerOutput(is_full_batch=last_sample_in_update, idx=main_idx)

                if not (state.is_full_update(self.batch_size) or state.is_full_epoch(self.samples_per_epoch)):
                    continue

                state.next_update(self.samples_per_epoch)

                for config, index_offset in zip(self.extra_samplers, self.index_offsets, strict=True):
                    if self._sampler_needs_to_iterate(state, config):
                        _logger.debug(
                            f"Interleaving sampler {config} at epoch {state.epoch}, update {state.update}, sample {state.sample}"
                        )
                        yield from self._iterate_from_sampler(config, index_offset)

                state.sample_at_last_update = state.sample
                if self._end_reached(state):
                    return
                # if drop_last -> skip last non-full batch
                if state.sample_in_epoch == self.samples_per_epoch:
                    break
