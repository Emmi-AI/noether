#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import numpy as np
import pytest

from noether.data import Dataset
from noether.data.base.wrappers import RepeatWrapper
from noether.data.base.wrappers.repeat import RepeatWrapperConfig


class MockDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def test_repeat_wrapper_initialization_and_length():
    """Tests that the wrapper correctly initializes and has the expected length."""
    original_dataset = MockDataset([1, 2, 3])
    repetitions = 3
    config = RepeatWrapperConfig(repetitions=repetitions, kind="")
    wrapper = RepeatWrapper(dataset=original_dataset, config=config)

    assert len(wrapper) == len(original_dataset) * repetitions
    assert wrapper.repetitions == repetitions


def test_repeat_wrapper_indices_and_getitem():
    """Tests that the indices are correctly tiled and items are fetched accordingly."""
    original_data = ["a", "b"]
    original_dataset = MockDataset(original_data)
    repetitions = 4
    config = RepeatWrapperConfig(repetitions=repetitions, kind="")
    wrapper = RepeatWrapper(dataset=original_dataset, config=config)

    # Expected length is 2 * 4 = 8
    assert len(wrapper) == 8

    # Expected indices are [0, 1, 0, 1, 0, 1, 0, 1]
    expected_indices = np.tile(np.arange(2), repetitions)
    np.testing.assert_array_equal(wrapper.indices, expected_indices)

    # Test item access
    assert wrapper[0] == original_data[0]
    assert wrapper[1] == original_data[1]
    assert wrapper[2] == original_data[0]
    assert wrapper[3] == original_data[1]
    assert wrapper[4] == original_data[0]
    assert wrapper[5] == original_data[1]
    assert wrapper[6] == original_data[0]
    assert wrapper[7] == original_data[1]


def test_repeat_wrapper_with_empty_dataset_raises_error():
    """Tests that initializing with an empty dataset raises a ValueError."""
    empty_dataset = MockDataset([])
    config = RepeatWrapperConfig(repetitions=5, kind="")
    with pytest.raises(ValueError, match="The dataset is empty."):
        RepeatWrapper(dataset=empty_dataset, config=config)


@pytest.mark.parametrize("repetitions", [1, 0, -1])
def test_repeat_wrapper_with_invalid_repetitions_raises_error(repetitions):
    """Tests that initializing with repetitions < 2 raises a ValueError."""
    original_dataset = MockDataset([1, 2, 3])
    with pytest.raises(
        ValueError,
    ):
        config = RepeatWrapperConfig(repetitions=repetitions, kind="")
        RepeatWrapper(dataset=original_dataset, config=config)


def test_repeat_wrapper_with_minimum_repetitions():
    """Tests the wrapper with the minimum allowed number of repetitions (2)."""
    original_dataset = MockDataset([10, 20])
    repetitions = 2
    config = RepeatWrapperConfig(repetitions=repetitions, kind="")
    wrapper = RepeatWrapper(dataset=original_dataset, config=config)

    assert len(wrapper) == 4
    assert wrapper[0] == 10
    assert wrapper[1] == 20
    assert wrapper[2] == 10
    assert wrapper[3] == 20
