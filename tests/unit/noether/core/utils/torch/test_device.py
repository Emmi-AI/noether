#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from unittest.mock import MagicMock

import pytest

from noether.core.utils.torch.device import move_items_to_device


@pytest.fixture
def mock_device():
    """Returns a dummy device object (e.g., 'cuda:0' or a Mock)."""
    return "cuda:0"


@pytest.fixture
def mock_tensor():
    """Returns a mock object acting like a Tensor with a .to() method."""
    mock = MagicMock()
    # Ensure .to() returns a 'moved' version of itself (or a new mock)
    mock.to.return_value = "moved_tensor"
    return mock


def test_move_single_item(mock_device, mock_tensor):
    """Test moving a standard item (Tensor) in the batch."""
    batch = {"data": mock_tensor}

    result = move_items_to_device(mock_device, batch)

    # Verify .to() was called correctly:
    mock_tensor.to.assert_called_with(mock_device, non_blocking=True)

    # Verify the batch was updated with the return value of .to():
    assert result["data"] == "moved_tensor"


def test_move_list_item_unwrapping(mock_device, mock_tensor):
    """Test that a list containing a single item is unwrapped and moved."""
    batch = {"data": [mock_tensor]}

    result = move_items_to_device(mock_device, batch)

    # Verify .to() was called on the item inside the list:
    mock_tensor.to.assert_called_with(mock_device, non_blocking=True)

    # Verify the list was removed and replaced by the item:
    assert result["data"] == "moved_tensor"
    assert not isinstance(result["data"], list)


def test_move_none_value(mock_device):
    """Test that None values are preserved and don't crash."""
    batch = {"missing": None}
    result = move_items_to_device(mock_device, batch)
    assert result["missing"] is None


def test_move_list_with_none(mock_device):
    """Test that a list containing [None] is unwrapped to None."""
    batch = {"missing_wrapped": [None]}
    result = move_items_to_device(mock_device, batch)
    assert result["missing_wrapped"] is None


def test_move_list_assertion_error_empty(mock_device):
    """Test that empty lists raise AssertionError (len != 1)."""
    batch = {"empty_list": list()}
    with pytest.raises(AssertionError):
        move_items_to_device(mock_device, batch)


def test_move_list_assertion_error_multiple(mock_device):
    """Test that lists with >1 items raise AssertionError."""
    batch = {"multi_list": [1, 2]}

    with pytest.raises(AssertionError):
        move_items_to_device(mock_device, batch)


def test_mixed_batch(mock_device, mock_tensor):
    """Test a complex batch with mixed types."""
    batch = {"tensor": mock_tensor, "wrapped_tensor": [mock_tensor], "none_val": None, "wrapped_none": [None]}
    result = move_items_to_device(mock_device, batch)

    assert result["tensor"] == "moved_tensor"
    assert result["wrapped_tensor"] == "moved_tensor"
    assert result["none_val"] is None
    assert result["wrapped_none"] is None
