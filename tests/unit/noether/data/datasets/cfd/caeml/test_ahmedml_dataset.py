#  Copyright © 2026 Emmi AI GmbH. All rights reserved.


import pytest
from pydantic import ValidationError

from noether.core.schemas.dataset import StandardDatasetConfig
from noether.data.datasets.cfd.caeml.ahmedml.dataset import AhmedMLDataset


def test_dataset_config_valid_minimal() -> None:
    """Test that a minimal valid config works."""
    config_data = {"kind": "ahmed_ml", "split": "train", "root": "/tmp/data"}
    config = StandardDatasetConfig(**config_data)
    assert config.kind == "ahmed_ml"
    assert config.split == "train"
    assert config.root == "/tmp/data"


def test_dataset_config_invalid_split() -> None:
    """Test that providing an invalid split name raises an error."""
    config_data = {
        "kind": "ahmed_ml",
        "split": "validation",  # valid options are 'train', 'val', 'test'
    }
    with pytest.raises(ValidationError) as exc_info:
        StandardDatasetConfig(**config_data)

    assert "Input should be 'train', 'val' or 'test'" in str(exc_info.value)


def test_dataset_config_forbids_extra_fields() -> None:
    """Test that 'extra' fields are forbidden as per model_config."""
    config_data = {
        "kind": "ahmed_ml",
        "split": "test",
        "random_field": 123,  # this should trigger an error
    }
    with pytest.raises(ValidationError) as exc_info:
        StandardDatasetConfig(**config_data)

    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_ahmedml_dataset_initialization(tmp_path) -> None:
    """
    Test that the AhmedMLDataset class initializes correctly.
    Uses 'tmp_path' fixture to provide a real, existing directory.
    """
    # 1. Arrange: Use tmp_path (converted to string) as the root
    config = StandardDatasetConfig(
        kind="ahmed_ml",
        root=str(tmp_path),
        split="train",
        pipeline=None,
    )

    dataset = AhmedMLDataset(dataset_config=config)

    assert dataset.config.root == str(tmp_path)
