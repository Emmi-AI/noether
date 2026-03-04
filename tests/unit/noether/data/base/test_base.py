#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest

from noether.core.schemas.dataset import DatasetBaseConfig
from noether.data import Dataset
from noether.data.base.dataset import with_normalizers
from noether.data.pipeline import Collator, MultiStagePipeline


class IndexDataset(Dataset):
    def __init__(self, size: int):
        super().__init__(dataset_config=DatasetBaseConfig(kind="index"))
        self.size = size
        self.indices = list(range(size))

    def getitem_x(self, idx: int):
        return self.indices[idx]

    def __len__(self) -> int:
        return self.size


@pytest.fixture
def index_dataset() -> IndexDataset:
    """Fixture for a simple dataset of size 3."""
    return IndexDataset(size=3)


def test_getitem(index_dataset: IndexDataset):
    """Test the __getitem__ method for a single item."""
    sample = index_dataset[0]
    assert isinstance(sample, dict)
    assert len(sample) == 2
    assert sample["index"] == 0
    assert sample["x"] == 0

    sample = index_dataset[1]
    assert sample["index"] == 1
    assert sample["x"] == 1

    assert index_dataset.pipeline is None  # Default collator should be None initially


def test_iter(index_dataset: IndexDataset):
    """Test iterating over the dataset."""
    samples = list(index_dataset)
    assert len(samples) == 3
    assert samples[0] == {"index": 0, "x": 0}
    assert samples[1] == {"index": 1, "x": 1}
    assert samples[2] == {"index": 2, "x": 2}


def test_len(index_dataset: IndexDataset):
    """Test the __len__ method."""
    assert len(index_dataset) == 3


def test_len_not_implemented_raises_error():
    """Test that NotImplementedError is raised if __len__ is not implemented."""

    class NoLenDataset(Dataset):
        pass

    with pytest.raises(NotImplementedError, match="__len__ method must be implemented"):
        len(NoLenDataset(dataset_config=DatasetBaseConfig(kind="train")))


def test_multiple_getitem_methods():
    """Test dataset with multiple getitem_* methods."""

    class MultiGetItemDataset(Dataset):
        def __init__(self):
            super().__init__(dataset_config=DatasetBaseConfig(kind="multi"))

        def __len__(self) -> int:
            return 1

        def getitem_x(self, idx: int) -> str:
            return "value_x"

        def getitem_y(self, idx: int) -> str:
            return "value_y"

    ds = MultiGetItemDataset()
    sample = ds[0]
    assert sample == {"index": 0, "x": "value_x", "y": "value_y"}


def test_collator_property(index_dataset: IndexDataset):
    """Test the getter and setter for the collator property."""
    assert index_dataset.pipeline is None

    new_collator = Collator()
    index_dataset.pipeline = new_collator
    assert index_dataset.pipeline is new_collator

    ms_collator = MultiStagePipeline(collators=[Collator()])
    index_dataset.pipeline = ms_collator
    assert index_dataset.pipeline is ms_collator

    with pytest.raises(TypeError):
        index_dataset.pipeline = "not a collator"


def test_iterator_over_dataset():
    dataset = IndexDataset(size=5)
    samples = list(dataset)
    assert len(samples) == 5
    for i, sample in enumerate(samples):
        assert sample == {"index": i, "x": i}
    assert i == 4  # Ensure the loop ran 5 times (0 to 4)


def test_with_normalizers_decorator_key_error():
    """Test that a KeyError is raised for a non-existent normalizer key."""

    class NormalizedDataset(Dataset):
        def __init__(self):
            super().__init__(dataset_config=DatasetBaseConfig(kind="normalized"))

        def __len__(self) -> int:
            return 1

        @with_normalizers("non_existent_key")
        def getitem_x(self, idx: int) -> str:
            return "raw_data"

    ds = NormalizedDataset()
    with pytest.raises(KeyError, match="Normalizer key 'non_existent_key' not found"):
        _ = ds[0]
