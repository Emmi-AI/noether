#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest

from noether.core.schemas.dataset import RepeatWrapperConfig
from noether.data import Dataset
from noether.data.base.wrappers import PropertySubsetWrapper, RepeatWrapper


class MockDataset(Dataset):
    """A mock dataset for testing purposes."""

    def __init__(self, length: int = 10):
        self._length = length
        self.some_property = "some_value"

    def __len__(self) -> int:
        return self._length

    def getitem_x(self, idx: int) -> str:
        return f"x_{idx}"

    def getitem_y(self, idx: int) -> str:
        return f"y_{idx}"

    def getitem_z(self, idx: int) -> str:
        return f"z_{idx}"

    def __getitem__(self, idx: int):
        # This is not used by ModeWrapper but required by the abstract base class
        raise NotImplementedError


@pytest.fixture
def mock_dataset() -> MockDataset:
    """Fixture for a mock dataset instance."""
    return MockDataset(length=20)


def test_mode_wrapper_init_success(mock_dataset):
    """Test successful initialization of ModeWrapper."""
    modes = {"x", "y"}
    wrapper = PropertySubsetWrapper(dataset=mock_dataset, properties=modes)

    assert wrapper.dataset is mock_dataset
    assert wrapper.properties == modes
    assert "x" in wrapper._getitem_functions
    assert "y" in wrapper._getitem_functions
    assert wrapper._getitem_functions["x"] == mock_dataset.getitem_x
    assert wrapper._getitem_functions["y"] == mock_dataset.getitem_y


def test_mode_wrapper_init_with_index_mode(mock_dataset):
    """Test initialization with the special 'index' mode."""
    modes = {"x", "index"}
    wrapper = PropertySubsetWrapper(dataset=mock_dataset, properties=modes)

    assert "index" in wrapper._getitem_functions
    assert wrapper._getitem_functions["index"](5) == 5


def test_mode_wrapper_init_raises_type_error_for_non_set_modes(mock_dataset):
    """Test that a TypeError is raised if modes is not a set."""
    with pytest.raises(TypeError):
        PropertySubsetWrapper(dataset=mock_dataset, properties=["x", "y"])  # type: ignore


def test_mode_wrapper_init_raises_attribute_error_for_missing_method(mock_dataset):
    """Test that an AttributeError is raised for a mode with no corresponding getitem method."""
    modes = {"x", "non_existent"}
    with pytest.raises(AttributeError, match=f"{type(mock_dataset)} has no method getitem_non_existent"):
        PropertySubsetWrapper(dataset=mock_dataset, properties=modes)


def test_getitem_success(mock_dataset):
    """Test the __getitem__ method for successful data retrieval."""
    modes = {"x", "y", "z", "index"}
    wrapper = PropertySubsetWrapper(dataset=mock_dataset, properties=modes)
    idx = 7
    expected_item = {
        "x": f"x_{idx}",
        "y": f"y_{idx}",
        "z": f"z_{idx}",
        "index": idx,
    }
    assert wrapper[idx] == expected_item


def test_getitem_raises_type_error_for_non_int_index(mock_dataset):
    """Test that __getitem__ raises a TypeError for a non-integer index."""
    wrapper = PropertySubsetWrapper(dataset=mock_dataset, properties={"z"})
    with pytest.raises(TypeError, match="Index must be an integer, got <class 'str'>."):
        wrapper["a"]  # type: ignore


def test_getitem_raises_value_error_for_negative_index(mock_dataset):
    """Test that __getitem__ raises a ValueError for a negative index."""
    wrapper = PropertySubsetWrapper(dataset=mock_dataset, properties={"x"})
    with pytest.raises(ValueError, match="Index must be non-negative, got -1."):
        wrapper[-1]


def test_getitem_raises_value_error_for_out_of_range_index(mock_dataset):
    """Test that __getitem__ raises a ValueError for an out-of-range index."""
    wrapper = PropertySubsetWrapper(dataset=mock_dataset, properties={"x"})
    with pytest.raises(IndexError, match="Index 21 is out of bounds for dataset of size 20."):
        wrapper[21]


def test_getattr_proxies_to_wrapped_dataset(mock_dataset):
    """Test that __getattr__ correctly proxies attribute access to the wrapped dataset."""
    wrapper = PropertySubsetWrapper(dataset=mock_dataset, properties={"x"})
    assert wrapper.some_property == "some_value"
    assert wrapper.__len__() == 20
    assert len(wrapper) == 20


def test_getattr_for_dataset_attribute(mock_dataset):
    """Test that accessing the 'dataset' attribute works correctly."""
    wrapper = PropertySubsetWrapper(dataset=mock_dataset, properties={"y"})
    assert wrapper.dataset is mock_dataset


def test_getattr_for_getitems_returns_none(mock_dataset):
    """Test that __getattr__ returns None for '__getitems__' to handle PyTorch DataLoader behavior."""
    wrapper = PropertySubsetWrapper(dataset=mock_dataset, properties={"y"})
    assert wrapper.__getitems__ is None


def test_getattr_raises_attribute_error_for_missing_attribute(mock_dataset):
    """Test that __getattr__ raises an AttributeError for a non-existent attribute."""
    wrapper = PropertySubsetWrapper(dataset=mock_dataset, properties={"x"})
    with pytest.raises(AttributeError):
        _ = wrapper.non_existent_property


def test_mode_wrapper_with_none_existing_mode(mock_dataset):
    """Test that initializing ModeWrapper with a non-existing mode raises an AttributeError."""
    with pytest.raises(AttributeError, match="Make sure the dataset implements getitem_<mode> for all modes."):
        PropertySubsetWrapper(dataset=mock_dataset, properties={"a"})


def test_double_wrapping_mode_wrapper(mock_dataset):
    """Test that wrapping a ModeWrapper with another ModeWrapper works correctly."""
    first_wrapper = PropertySubsetWrapper(dataset=mock_dataset, properties={"x", "y"})
    second_wrapper = RepeatWrapper(dataset=first_wrapper, config=RepeatWrapperConfig(repetitions=2, kind=""))

    idx = 3
    expected_item = {
        "y": f"y_{idx}",
        "x": f"x_{idx}",
    }
    assert len(second_wrapper) == len(first_wrapper) * 2
    assert second_wrapper[idx] == expected_item

    for i in range(len(second_wrapper)):
        original_idx = i % len(first_wrapper)
        expected_item = {
            "y": f"y_{original_idx}",
            "x": f"x_{original_idx}",
        }
        assert second_wrapper[i] == expected_item


def test_double_property_wrapper(mock_dataset):
    """Test that wrapping a PropertySubsetWrapper with another PropertySubsetWrapper works correctly."""
    first_wrapper = PropertySubsetWrapper(dataset=mock_dataset, properties={"x", "y"})
    second_wrapper = PropertySubsetWrapper(dataset=first_wrapper, properties={"y"})

    idx = 5
    expected_item = {
        "y": f"y_{idx}",
    }
    assert len(second_wrapper) == len(first_wrapper)
    assert second_wrapper[idx] == expected_item

    for i in range(len(second_wrapper)):
        original_idx = i
        expected_item = {
            "y": f"y_{original_idx}",
        }
        assert second_wrapper[i] == expected_item


def test_double_wrapping_mode_wrapper_2(mock_dataset):
    """Test that wrapping a ModeWrapper with another ModeWrapper works correctly."""

    first_wrapper = RepeatWrapper(dataset=mock_dataset, config=RepeatWrapperConfig(repetitions=2, kind=""))
    second_wrapper = PropertySubsetWrapper(dataset=first_wrapper, properties={"x", "y"})

    idx = 3
    expected_item = {
        "y": f"y_{idx}",
        "x": f"x_{idx}",
    }
    assert len(second_wrapper) == len(first_wrapper)
    assert second_wrapper[idx] == expected_item

    for i in range(len(second_wrapper)):
        second_wrapper[i]


def test_double_property_wrapper_2(mock_dataset):
    """Test that wrapping a PropertySubsetWrapper with another PropertySubsetWrapper works correctly."""
    first_wrapper = PropertySubsetWrapper(dataset=mock_dataset, properties={"x", "y"})
    second_wrapper = PropertySubsetWrapper(dataset=first_wrapper, properties={"y"})

    idx = 5
    expected_item = {
        "y": f"y_{idx}",
    }
    assert len(second_wrapper) == len(first_wrapper)
    assert second_wrapper[idx] == expected_item

    for i in range(len(second_wrapper)):
        original_idx = i
        expected_item = {
            "y": f"y_{original_idx}",
        }
        assert second_wrapper[i] == expected_item
        assert first_wrapper[i]["x"] == f"x_{original_idx}"
        assert first_wrapper[i]["y"] == f"y_{original_idx}"
