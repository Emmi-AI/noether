#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from collections import Counter

import pytest

from noether.core.factory.dataset import DatasetFactory
from noether.core.schemas.dataset import (
    RepeatWrapperConfig,
    ShuffleWrapperConfig,
    SubsetWrapperConfig,
)
from noether.data.base.wrapper import DatasetWrapper
from tests.test_training_pipeline.dummy_project.schemas.datasets.base_dataset_config import (
    BaseDatasetConfig,
)


@pytest.mark.parametrize(
    (
        "wrappers",
        "expected_length",
        "expected_unique_indices",
        "repeat_factor",
    ),
    [
        pytest.param(
            [ShuffleWrapperConfig(kind="noether.data.base.wrappers.shuffle.ShuffleWrapper", seed=123)],
            12,
            12,
            1,
            id="shuffle-only",
        ),
        pytest.param(
            [
                SubsetWrapperConfig(
                    kind="noether.data.base.wrappers.subset.SubsetWrapper",
                    start_percent=0,
                    end_percent=0.5,
                )
            ],
            6,
            6,
            1,
            id="subset-only",
        ),
        pytest.param(
            [RepeatWrapperConfig(kind="noether.data.base.wrappers.repeat.RepeatWrapper", repetitions=3)],
            36,
            12,
            3,
            id="repeat-only",
        ),
        pytest.param(
            [
                ShuffleWrapperConfig(kind="noether.data.base.wrappers.shuffle.ShuffleWrapper", seed=42),
                SubsetWrapperConfig(
                    kind="noether.data.base.wrappers.subset.SubsetWrapper",
                    start_index=0,
                    end_index=6,
                ),
                RepeatWrapperConfig(kind="noether.data.base.wrappers.repeat.RepeatWrapper", repetitions=2),
            ],
            12,
            6,
            2,
            id="all-wrappers",
        ),
    ],
)
def test_dataset_factory_applies_wrappers(
    wrappers: list[ShuffleWrapperConfig | SubsetWrapperConfig | RepeatWrapperConfig],
    expected_length: int,
    expected_unique_indices: int,
    repeat_factor: int,
) -> None:
    base_config = BaseDatasetConfig(
        kind="tests.test_training_pipeline.dummy_project.datasets.base_dataset.BaseDataset",
        num_samples=12,
        num_classes=3,
        noise=0.0,
        radius=1.0,
        dataset_wrappers=wrappers,
    )

    dataset = DatasetFactory().create(base_config)

    # Check that the dataset is a DatasetWrapper and has the expected length
    assert isinstance(dataset, DatasetWrapper)
    assert len(dataset) == expected_length

    samples = [dataset[i] for i in range(len(dataset))]
    indices = [sample["index"] for sample in samples]

    # Check that the number of unique indices matches the expected value
    assert len(set(indices)) == expected_unique_indices

    # Check that each unique index is repeated the expected number of times
    counts = Counter(indices)
    assert all(count == repeat_factor for count in counts.values())

    # Check that the samples contain the expected keys
    first_sample = samples[0]
    assert {"index", "x", "y"}.issubset(first_sample.keys())
