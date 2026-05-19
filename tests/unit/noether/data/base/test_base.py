#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest

from noether.data import Dataset
from noether.data.base.dataset import DatasetBaseConfig, with_normalizers
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
    """Test that that the standard behavior of with_normalizers is preserved when a non-existent key is used."""

    class NormalizedDataset(Dataset):
        def __init__(self):
            super().__init__(dataset_config=DatasetBaseConfig(kind="normalized"))

        def __len__(self) -> int:
            return 1

        @with_normalizers("non_existent_key")
        def getitem_x(self, idx: int) -> str:
            return "raw_data"

    ds = NormalizedDataset()
    out = ds[0]
    assert out == {"index": 0, "x": "raw_data"}


# --- pre_getitem hook tests ---


def test_pre_getitem_default_returns_none(index_dataset: IndexDataset):
    """Default pre_getitem returns {} and getitem still works normally."""
    assert index_dataset.pre_getitem(0) == {}
    sample = index_dataset[0]
    assert sample == {"index": 0, "x": 0}


def test_pre_getitem_forwards_kwargs():
    """pre_getitem dict is forwarded as kwargs to getitem methods that accept them."""

    class SharedLoadDataset(Dataset):
        def __init__(self):
            super().__init__(dataset_config=DatasetBaseConfig(kind="shared"))
            self.data = {0: {"a": 10, "b": 20}}

        def __len__(self) -> int:
            return 1

        def pre_getitem(self, idx: int) -> dict:
            return self.data[idx]

        def getitem_a(self, idx: int, *, a: int = 0, **kwargs) -> int:
            return a

        def getitem_b(self, idx: int, *, b: int = 0, **kwargs) -> int:
            return b

    ds = SharedLoadDataset()
    sample = ds[0]
    assert sample == {"index": 0, "a": 10, "b": 20}


def test_pre_getitem_skips_methods_without_extra_params():
    """Getitem methods that only take idx are called without extra kwargs."""

    class MixedDataset(Dataset):
        def __init__(self):
            super().__init__(dataset_config=DatasetBaseConfig(kind="mixed"))

        def __len__(self) -> int:
            return 1

        def pre_getitem(self, idx: int) -> dict:
            return {"shared_value": 42}

        def getitem_plain(self, idx: int) -> str:
            return "plain"

        def getitem_fancy(self, idx: int, *, shared_value: int = 0, **kwargs) -> int:
            return shared_value

    ds = MixedDataset()
    sample = ds[0]
    assert sample["plain"] == "plain"
    assert sample["fancy"] == 42


def test_post_getitem_called_after_getitem(index_dataset: IndexDataset):
    """post_getitem is called after all getitem_* methods."""
    calls = []
    original_getitem_x = index_dataset.getitem_x

    def tracked_getitem_x(idx):
        calls.append("getitem_x")
        return original_getitem_x(idx)

    def tracked_post(idx, pre):
        calls.append("post_getitem")

    index_dataset.getitem_x = tracked_getitem_x
    index_dataset.post_getitem = tracked_post
    index_dataset[0]
    assert calls == ["getitem_x", "post_getitem"]


def test_post_getitem_receives_pre_value():
    """post_getitem receives the same value that pre_getitem returned."""
    received_pre = []

    class TrackingDataset(Dataset):
        def __init__(self):
            super().__init__(dataset_config=DatasetBaseConfig(kind="tracking"))

        def __len__(self) -> int:
            return 1

        def pre_getitem(self, idx: int) -> dict:
            return {"handle": "open_resource"}

        def post_getitem(self, idx: int, pre: dict | None) -> None:
            received_pre.append(pre)

        def getitem_x(self, idx: int) -> str:
            return "data"

    ds = TrackingDataset()
    ds[0]
    assert received_pre == [{"handle": "open_resource"}]


def test_post_getitem_called_on_error():
    """post_getitem runs even when a getitem_* method raises."""
    cleanup_called = []

    class FailingDataset(Dataset):
        def __init__(self):
            super().__init__(dataset_config=DatasetBaseConfig(kind="failing"))

        def __len__(self) -> int:
            return 1

        def pre_getitem(self, idx: int) -> dict:
            return {"fd": 99}

        def post_getitem(self, idx: int, pre: dict | None) -> None:
            cleanup_called.append(pre)

        def getitem_x(self, idx: int) -> str:
            raise RuntimeError("broken")

    ds = FailingDataset()
    with pytest.raises(RuntimeError, match="broken"):
        ds[0]
    assert cleanup_called == [{"fd": 99}]
