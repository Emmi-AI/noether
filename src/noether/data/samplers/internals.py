#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import bisect
from collections.abc import Iterable
from typing import NamedTuple

from torch.utils.data import ConcatDataset


# can't be a local class as it is required to be pickleable
class _InterleavedConcatDataset(ConcatDataset):
    """`torch.utils.data.ConcatDataset` but it returns the dataset index."""

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, self.datasets[dataset_idx][sample_idx]


# can't be a local class as it is required to be pickleable
class _InterleavedCollator:
    """Holds a list of collators and calls a single one of them based on the dataset_idx that was returned."""

    def __init__(self, collators):
        self.collators = collators

    def __call__(self, data):
        dataset_idxs, data = zip(*data, strict=True)
        if not all(dataset_idxs[0] == idx for idx in dataset_idxs):
            raise ValueError("All samples in a batch must come from the same dataset.")
        return self.collators[dataset_idxs[0]](data)


class SamplerOutput(NamedTuple):  #  lint-ignore: NoNamedTuple
    """Output of the InterleavedSampler. Contains the sample index and the dataset index."""

    is_full_batch: bool
    idx: int


# can't be a local class as it is required to be pickleable
class _InterleavedBatchSampler:
    """Creates batches of indices from an iterable of indices."""

    def __init__(self, sampler: Iterable[SamplerOutput]):
        super().__init__()
        self.sampler = sampler

    def __iter__(self):
        idxs = []
        for is_full_batch, idx in self.sampler:
            idxs.append(idx)
            if is_full_batch:
                yield idxs
                idxs = []
        assert len(idxs) == 0

    def __len__(self):
        raise NotImplementedError
