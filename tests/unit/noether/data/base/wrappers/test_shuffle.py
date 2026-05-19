#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import numpy as np
import pytest
from pydantic import ValidationError

from noether.data import Dataset
from noether.data.base.wrappers import ShuffleWrapper
from noether.data.base.wrappers.shuffle import ShuffleWrapperConfig


class MockDataset(Dataset):
    def __init__(self, size=10):
        self._data = list(range(size))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


def test_shuffle_wrapper_initialization_and_length():
    """Tests that the wrapper correctly initializes and has the expected length."""
    original_dataset = MockDataset(size=20)
    config = ShuffleWrapperConfig(seed=42, kind="")
    wrapper = ShuffleWrapper(dataset=original_dataset, config=config)

    assert len(wrapper) == len(original_dataset)
    assert isinstance(wrapper.indices, np.ndarray)
    assert set(wrapper.indices) == set(range(len(original_dataset)))
    np.testing.assert_array_equal(wrapper.indices, np.random.default_rng(42).permutation(len(original_dataset)))


def test_wrong_configs():
    """Tests that invalid configurations raise appropriate errors."""
    original_dataset = MockDataset(size=10)

    # Negative seed
    with pytest.raises(ValidationError):
        config = ShuffleWrapperConfig(seed=-1, kind="")
        # ShuffleWrapper(dataset=original_dataset, config=config)

    # Non-integer seed
    with pytest.raises(ValidationError):
        config = ShuffleWrapperConfig(seed=3.5, kind="")
        # ShuffleWrapper(dataset=original_dataset, config=config)
