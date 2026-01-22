#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import functools
import logging

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Sampler, SequentialSampler

from noether.core.distributed import is_distributed
from noether.core.schemas.dataset import ShuffleWrapperConfig, SubsetWrapperConfig
from noether.core.utils.common.stopwatch import Stopwatch
from noether.core.utils.platform import get_fair_cpu_count, get_total_cpu_count
from noether.data import Dataset
from noether.data.base.wrapper import DatasetWrapper
from noether.data.base.wrappers import (
    META_GETITEM_TIME,
    PropertySubsetWrapper,
    ShuffleWrapper,
    SubsetWrapper,
    TimingWrapper,
)
from noether.data.pipeline.collator import CollatorType
from noether.data.samplers import InterleavedSampler, SamplerIntervalConfig
from noether.data.samplers.interleaved_sampler import InterleavedSamplerConfig

META_GETITEM_COLLATOR = "__meta_time_collate"


def _timing_collate_fn(collator, batch):
    time_getitem = None
    if len(batch) > 0 and META_GETITEM_TIME in batch[0][1]:
        time_getitem = sum([sample[1].pop(META_GETITEM_TIME) for sample in batch])
    with Stopwatch() as sw:
        res = collator(batch)
    if isinstance(res, dict):
        res[META_GETITEM_COLLATOR] = sw.elapsed_seconds
        if time_getitem is not None:
            res[META_GETITEM_TIME] = time_getitem
    return res


class DataContainer:
    """Container that holds datasets and provides utilities for datasets and data loading."""

    def __init__(self, datasets: dict[str, Dataset], num_workers: int | None = None, pin_memory: bool = True):
        """
        Args:
            datasets: A dictionary with datasets for the training run.
            num_workers: Number of data loading workers to use. If None, will use (#CPUs / #GPUs - 1) workers.
                The `-1` keeps 1 CPU free for the main process. Defaults to None.
            pin_memory: Is passed as `pin_memory` to `torch.utils.data.DataLoader`. Defaults to True.
        """
        self.logger = logging.getLogger(type(self).__name__)
        if len(datasets) == 0:
            raise ValueError("At least one dataset must be provided to DataContainer.")
        self.datasets = datasets
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # set first dataset as "train" dataset in place of an actual dataset
        # the "train" dataset is used to propagate shapes, so in evaluation runs if no train dataset is present,
        # the first evaluation dataset is used instead

        if "train" not in self.datasets:
            self.datasets["train"] = next(iter(self.datasets.values()))

    def get_dataset(
        self,
        key: str | None = None,
        properties: set[str] | None = None,
        max_size: int | None = None,
        shuffle_seed: int | None = None,
    ) -> Dataset | DatasetWrapper:
        """Returns the dataset identified by key (or the first dataset if no key is provided) with optional wrapping
        into a :class:`ShuffleWrapper` (via `shuffle_seed`), a :class:`SubsetWrapper` (via `max_size`) or a
        :class:`PropertySubsetWrapper`. Note that the wrappers can be used at once or individually, in case when all
        arguments are provided the order will be:

            Dataset -> ShuffleWrapper(Optional) -> SubsetWrapper(Optional) -> PropertySubsetWrapper(Optional)

        Args:
            key: Identifier of the dataset. If None, returns the first dataset of the `DataContainer`. Defaults to None.
            properties: If defined, overrides the properties to load from the dataset. If not defined, uses the
                properties defined in the dataset itself or all properties if none are defined.
            max_size: If defined, wraps the dataset into a SubsetWrapper with the specified `max_size`.
                Default: None (no wrapping)
            shuffle_seed: If defined, wraps the dataset into a ShuffleWrapper with the specified
                `shuffle_seed`. Defaults to None (=no wrapping).

        Returns:
            Dataset: Dataset of the DataContainer optionally wrapped into dataset wrappers.
        """
        key = key or next(iter(self.datasets.keys()))
        dataset: Dataset = self.datasets[key]
        if shuffle_seed is not None:
            dataset = ShuffleWrapper(
                dataset=dataset,
                config=ShuffleWrapperConfig(kind="", seed=shuffle_seed),  # type: ignore[arg-type]
            )  # type: ignore  # FIXME: kind
        if max_size is not None:
            dataset = SubsetWrapper(dataset, config=SubsetWrapperConfig(kind="", end_index=max_size))  # type: ignore[assignment]
        if properties is not None:
            dataset = PropertySubsetWrapper(dataset=dataset, properties=properties)  # type: ignore[assignment]
        dataset = TimingWrapper(dataset=dataset)  # type: ignore[assignment]
        return dataset  # type: ignore

    def get_main_sampler(self, train_dataset: Dataset | DatasetWrapper, shuffle: bool = True) -> Sampler[int]:
        """Creates the `main_sampler` for data loading.

        Args:
            train_dataset: Dataset that is used for training.
            shuffle: Either or not to randomly shuffle the sampled indices before every epoch. Defaults to True.

        Returns:
            Sampler: Sampler to be used for sampling indices of the `train_dataset`.
        """
        if is_distributed():
            self.logger.info(f"Using DistributedSampler(shuffle={shuffle}) as main_sampler")
            # NOTE: drop_last is required as otherwise len(sampler) can be larger than len(dataset)
            # which results in unconsumed batches from InterleavedSampler
            return DistributedSampler(
                train_dataset,  # type: ignore[arg-type]
                shuffle=shuffle,
                drop_last=True,
            )
        if shuffle:
            self.logger.info("Using RandomSampler as main_sampler")
            return RandomSampler(train_dataset)
        else:
            self.logger.info("Using SequentialSampler as main_sampler")
            return SequentialSampler(train_dataset)

    def get_data_loader(
        self,
        train_sampler: Sampler,
        train_collator: CollatorType | None,
        batch_size: int,
        epochs: int | None,
        updates: int | None,
        samples: int | None,
        callback_samplers: list[SamplerIntervalConfig],
        start_epoch: int | None = None,
        evaluation: bool = False,
    ) -> torch.utils.data.DataLoader:
        """Creates a `torch.utils.data.DataLoader` that can be used for efficient data loading by utilizing an
        `InterleavedSampler` based on the `main_sampler`, `configs` and other arguments that are passed to this method.

        Args:
            train_sampler: Sampler to be used for the main dataset (i.e., training dataset).
            train_collator: Collator to collate samples from the main dataset (i.e., training dataset).
            batch_size: batch_size to use for training.
            epochs: For how many epochs does the training last.
            updates: For how many updates does the training last.
            samples: For how many samples does the training last.
            callback_samplers: List of SamplerIntervalConfigs to use for callback sampling.
            start_epoch: At which epoch to start (used for resuming training). Mutually exclusive with `start_update`
                and `start_sample`.

        Returns:
            DataLoader: Object from which data can be loaded according to the specified configuration.
        """
        sampler = InterleavedSampler(
            train_sampler=train_sampler,  # type: ignore
            callback_samplers=callback_samplers,
            train_collator=train_collator,
            config=InterleavedSamplerConfig(
                max_epochs=epochs,
                max_updates=updates,
                max_samples=samples,
                start_epoch=start_epoch,
                batch_size=batch_size,
                evaluation=evaluation,
            ),
        )
        if self.num_workers is None:
            num_workers = get_fair_cpu_count()
        else:
            num_workers = self.num_workers

        loader = DataLoader(
            dataset=sampler.dataset,
            batch_sampler=sampler.batch_sampler,
            collate_fn=functools.partial(_timing_collate_fn, sampler.collator),
            num_workers=num_workers,
            pin_memory=self.pin_memory,
        )

        self.logger.info(
            f"Created dataloader (batch_size={batch_size} num_workers={loader.num_workers} "
            f"pin_memory={loader.pin_memory} total_cpu_count={get_total_cpu_count()} "
            f"prefetch_factor={loader.prefetch_factor})"
        )
        return loader
