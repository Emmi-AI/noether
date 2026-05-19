#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import numpy as np
import pytest

from noether.data import Dataset
from noether.data.base.wrappers import SubsetWrapper
from noether.data.base.wrappers.subset import SubsetWrapperConfig


class MockDataset(Dataset):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


@pytest.fixture
def mock_dataset():
    return MockDataset(list(range(10)))


def test_subset_wrapper_with_indices(mock_dataset):
    config = SubsetWrapperConfig(indices=[1, 3, 5, 7], kind="")
    wrapper = SubsetWrapper(dataset=mock_dataset, config=config)
    assert len(wrapper) == 4
    assert wrapper[0] == 1
    assert wrapper[1] == 3
    assert wrapper[2] == 5
    assert wrapper[3] == 7
    assert list(wrapper) == [1, 3, 5, 7]


def test_subset_wrapper_with_start_index(mock_dataset):
    config = SubsetWrapperConfig(start_index=7, kind="")
    wrapper = SubsetWrapper(dataset=mock_dataset, config=config)
    assert len(wrapper) == 3
    assert list(wrapper) == [7, 8, 9]


def test_subset_wrapper_with_end_index(mock_dataset):
    config = SubsetWrapperConfig(end_index=3, kind="")
    wrapper = SubsetWrapper(dataset=mock_dataset, config=config)
    assert len(wrapper) == 3
    assert list(wrapper) == [0, 1, 2]


def test_subset_wrapper_with_start_and_end_index(mock_dataset):
    config = SubsetWrapperConfig(start_index=2, end_index=5, kind="")
    wrapper = SubsetWrapper(dataset=mock_dataset, config=config)
    assert len(wrapper) == 3
    assert list(wrapper) == [2, 3, 4]


def test_subset_wrapper_with_end_index_greater_than_len(mock_dataset):
    config = SubsetWrapperConfig(end_index=15, kind="")
    wrapper = SubsetWrapper(dataset=mock_dataset, config=config)
    assert len(wrapper) == 10
    assert list(wrapper) == list(range(10))


def test_subset_wrapper_with_start_percent(mock_dataset):
    config = SubsetWrapperConfig(start_percent=0.7, kind="")
    wrapper = SubsetWrapper(dataset=mock_dataset, config=config)
    assert len(wrapper) == 3
    assert list(wrapper) == [7, 8, 9]


def test_subset_wrapper_with_end_percent(mock_dataset):
    config = SubsetWrapperConfig(end_percent=0.3, kind="")
    wrapper = SubsetWrapper(dataset=mock_dataset, config=config)
    assert len(wrapper) == 3
    assert list(wrapper) == [0, 1, 2]


def test_subset_wrapper_with_start_and_end_percent(mock_dataset):
    config = SubsetWrapperConfig(start_percent=0.2, end_percent=0.5, kind="")
    wrapper = SubsetWrapper(dataset=mock_dataset, config=config)
    assert len(wrapper) == 3
    assert list(wrapper) == [2, 3, 4]


def test_init_raises_error_no_args(mock_dataset):
    config = SubsetWrapperConfig(kind="")
    with pytest.raises(RuntimeError, match="Either indices or start_index/end_index or start_percent/end_percent"):
        SubsetWrapper(dataset=mock_dataset, config=config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"indices": [1], "start_index": 0},
        {"indices": [1], "end_index": 1},
        {"indices": [1], "start_percent": 0.0},
        {"indices": [1], "end_percent": 1.0},
    ],
)
def test_init_raises_error_indices_with_other_args(mock_dataset, kwargs):
    with pytest.raises(
        ValueError, match="Cannot specify indices together with start_index/end_index or start_percent/end_percent."
    ):
        config = SubsetWrapperConfig(**kwargs, kind="")
        SubsetWrapper(dataset=mock_dataset, config=config)


def test_init_raises_error_index_and_percent_args(mock_dataset):
    config = SubsetWrapperConfig(start_index=0, start_percent=0.5, kind="")
    with pytest.raises(ValueError, match="Cannot specify both start_index/end_index and start_percent/end_percent."):
        SubsetWrapper(dataset=mock_dataset, config=config)


@pytest.mark.parametrize(
    ("kwargs", "error_msg"),
    [
        ({"start_index": 1.1}, "start_index and end_index must be of type int or None."),
        ({"end_index": "a"}, "start_index and end_index must be of type int or None."),
        ({"start_index": 5, "end_index": 2}, "start_index \\(5\\) must be less than or equal to end_index \\(2\\)."),
    ],
)
def test_init_raises_error_invalid_index_args(mock_dataset, kwargs, error_msg):
    with pytest.raises(ValueError):
        config = SubsetWrapperConfig(**kwargs, kind="")
        SubsetWrapper(dataset=mock_dataset, config=config)


@pytest.mark.parametrize(
    ("kwargs", "error_msg"),
    [
        ({"start_percent": -0.1}, "start_percent must be of type float and between 0.0 and 1.0."),
        ({"start_percent": 1.1}, "start_percent must be of type float and between 0.0 and 1.0."),
        ({"end_percent": -0.1}, "end_percent must be of type float and between 0.0 and 1.0."),
        ({"end_percent": 1.1}, "end_percent must be of type float and between 0.0 and 1.0."),
        ({"start_percent": "a"}, "start_percent must be of type float and between 0.0 and 1.0."),
        ({"end_percent": "b"}, "end_percent must be of type float and between 0.0 and 1.0."),
        (
            {"start_percent": 0.5, "end_percent": 0.2},
            "end_percent \\(0.2\\) must be larger than start_percent \\(0.5\\).",
        ),
        (
            {"start_percent": 0.5, "end_percent": 0.5},
            "end_percent \\(0.5\\) must be larger than start_percent \\(0.5\\).",
        ),
    ],
)
def test_init_raises_error_invalid_percent_args(mock_dataset, kwargs, error_msg):
    with pytest.raises(ValueError):
        config = SubsetWrapperConfig(**kwargs, kind="")
        SubsetWrapper(dataset=mock_dataset, config=config)


def test_subset_wrapper_indices_are_numpy_int64(mock_dataset):
    config = SubsetWrapperConfig(start_index=2, end_index=5, kind="")
    wrapper = SubsetWrapper(dataset=mock_dataset, config=config)
    assert isinstance(wrapper.indices, np.ndarray)
    assert wrapper.indices.dtype == np.int64
