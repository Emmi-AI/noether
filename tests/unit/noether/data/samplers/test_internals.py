#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import pytest
from torch.utils.data import Dataset

from noether.data.samplers.internals import (
    _InterleavedBatchSampler,
    _InterleavedCollator,
    _InterleavedConcatDataset,
)


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestInterleavedConcatDataset:
    def test_getitem_returns_index_and_data(self):
        ds1 = SimpleDataset(["a", "b"])
        ds2 = SimpleDataset(["c", "d", "e"])
        concat_ds = _InterleavedConcatDataset([ds1, ds2])

        assert concat_ds[0] == (0, "a")
        assert concat_ds[1] == (0, "b")

        assert concat_ds[2] == (1, "c")
        assert concat_ds[3] == (1, "d")
        assert concat_ds[4] == (1, "e")

    def test_negative_indexing(self):
        ds1 = SimpleDataset([1, 2])
        ds2 = SimpleDataset([3, 4])
        concat_ds = _InterleavedConcatDataset([ds1, ds2])

        assert concat_ds[-1] == (1, 4)  # last element
        assert concat_ds[-3] == (0, 2)  # second element of first dataset

    def test_index_out_of_bounds(self):
        ds1 = SimpleDataset([1])
        concat_ds = _InterleavedConcatDataset([ds1])

        with pytest.raises(IndexError):
            _ = concat_ds[10]

        with pytest.raises(ValueError, match="absolute value of index should not exceed"):
            _ = concat_ds[-10]


class TestInterleavedCollator:
    def test_delegates_to_correct_collator(self):
        collator_1 = lambda x: f"collator_1 processed {x}"
        collator_2 = lambda x: f"collator_2 processed {x}"

        interleaved_collator = _InterleavedCollator([collator_1, collator_2])

        batch_ds0 = [(0, "data_a"), (0, "data_b")]
        result_0 = interleaved_collator(batch_ds0)
        assert result_0 == "collator_1 processed ('data_a', 'data_b')"

        batch_ds1 = [(1, "data_x"), (1, "data_y")]
        result_1 = interleaved_collator(batch_ds1)
        assert result_1 == "collator_2 processed ('data_x', 'data_y')"

    def test_raises_mixed_batch_error(self):
        interleaved_collator = _InterleavedCollator([lambda x: x, lambda x: x])
        mixed_batch = [(0, "data_a"), (1, "data_x")]

        with pytest.raises(ValueError, match="All samples in a batch must come from the same dataset"):
            interleaved_collator(mixed_batch)


class TestInterleavedBatchSampler:
    def test_iter_yields_batches(self):
        # (is_full_batch_flag, index):
        mock_sampler_output = [
            (False, 0),
            (False, 1),
            (True, 2),  # batch 1 ends here: [0, 1, 2]
            (False, 3),
            (True, 4),  # batch 2 ends here: [3, 4]
        ]

        batch_sampler = _InterleavedBatchSampler(mock_sampler_output)

        # list() calls __len__ which raises NotImplementedError in this class, so this is a workaround for that:
        batches = [b for b in batch_sampler]  # noqa: C416

        assert len(batches) == 2
        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4]

    def test_asserts_empty_buffer_at_end(self):
        mock_sampler_output = [
            (False, 0),
            (False, 1),
            # Ends without a True flag
        ]
        batch_sampler = _InterleavedBatchSampler(mock_sampler_output)

        with pytest.raises(AssertionError):
            _ = [b for b in batch_sampler]  # noqa: C416

    def test_len_not_implemented(self):
        batch_sampler = _InterleavedBatchSampler([])
        with pytest.raises(NotImplementedError):
            len(batch_sampler)
