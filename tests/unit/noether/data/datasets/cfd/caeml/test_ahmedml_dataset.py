#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from noether.core.schemas.dataset import DatasetBaseConfig
from noether.data.datasets.cfd.caeml.ahmedml.dataset import AhmedMLDataset


def test_dataset_config_valid_minimal() -> None:
    """Test that a minimal valid config works."""
    config_data = {
        "kind": "ahmed_ml",
        "split": "train"
    }
    config = DatasetBaseConfig(**config_data)
    assert config.kind == "ahmed_ml"
    assert config.split == "train"
    assert config.root is None


def test_dataset_config_invalid_split() -> None:
    """Test that providing an invalid split name raises an error."""
    config_data = {
        "kind": "ahmed_ml",
        "split": "validation"  # valid options are 'train', 'val', 'test'
    }
    with pytest.raises(ValidationError) as exc_info:
        DatasetBaseConfig(**config_data)

    assert "Input should be 'train', 'val' or 'test'" in str(exc_info.value)


def test_dataset_config_forbids_extra_fields() -> None:
    """Test that 'extra' fields are forbidden as per model_config."""
    config_data = {
        "kind": "ahmed_ml",
        "split": "test",
        "random_field": 123,  # this should trigger an error
    }
    with pytest.raises(ValidationError) as exc_info:
        DatasetBaseConfig(**config_data)

    assert "Extra inputs are not permitted" in str(exc_info.value)


@pytest.fixture
def mock_pipeline():
    """Creates a dummy pipeline object to pass into the config."""
    return MagicMock()


def test_ahmedml_dataset_initialization(mock_pipeline, tmp_path) -> None:
    """
    Test that the AhmedMLDataset class initializes correctly.
    Uses 'tmp_path' fixture to provide a real, existing directory.
    """
    # 1. Arrange: Use tmp_path (converted to string) as the root
    config = DatasetBaseConfig(
        kind="ahmed_ml",
        root=str(tmp_path),
        split="train",
        pipeline=mock_pipeline,
    )

    dataset = AhmedMLDataset(dataset_config=config)

    assert dataset.config.root == str(tmp_path)
