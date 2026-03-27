#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.data.pipeline.sample_processors import PointSamplingSampleProcessor


@pytest.fixture
def sample_data():
    return [
        {
            "input_position": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "output_position": torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        },
        {
            "input_position": torch.tensor([[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]),
            "output_position": torch.tensor([[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]]),
        },
    ]


def test_point_sampling_precollator(sample_data):
    torch.manual_seed(42)
    items = {"input_position", "output_position"}
    num_points = 2
    precollator = PointSamplingSampleProcessor(items, num_points)

    for sample in sample_data:
        processed_sample = precollator(sample)

        assert len(processed_sample) == len(sample)

        for item in items:
            assert processed_sample[item].shape[0] == num_points
            assert torch.all(torch.isin(processed_sample[item], sample[item])), (
                "Subsampled points should exist in the original tensor"
            )


def test_point_sampling_preprocessor_one_item(sample_data):
    torch.manual_seed(42)
    items = {"input_position"}
    num_points = 2
    precollator = PointSamplingSampleProcessor(items, num_points)

    for sample in sample_data:
        processed_sample = precollator(sample)

        assert len(processed_sample) == len(sample)

        for item in items:
            assert processed_sample[item].shape[0] == num_points
            assert processed_sample["output_position"].shape[0] == 3
            assert torch.all(torch.isin(processed_sample[item], sample[item])), (
                "Subsampled points should exist in the original tensor"
            )


def test_point_sampling_precollator_nondeterministic():
    torch.manual_seed(0)
    sample = {"pos": torch.rand(10, 3)}
    precollator = PointSamplingSampleProcessor(items={"pos"}, num_points=4)

    processed1 = precollator(sample)
    processed2 = precollator(sample)

    assert len(processed1["pos"]) == len(processed2["pos"])
    assert not torch.equal(processed1["pos"], processed2["pos"])


def test_point_sampling_precollator_deterministic():
    sample = {"pos": torch.rand(10, 3), "index": 0}
    precollator = PointSamplingSampleProcessor(items={"pos"}, num_points=4, seed=0)

    processed1 = precollator(sample)
    processed2 = precollator(sample)

    assert len(processed1["pos"]) == len(processed2["pos"])
    assert torch.equal(processed1["pos"], processed2["pos"])


def test_point_sampling_precollator_invalid_num_points(sample_data):
    items = {"input_position", "output_position"}
    num_points = 0
    with pytest.raises(AssertionError):
        PointSamplingSampleProcessor(items, num_points)


def test_denormalize():
    precollator = PointSamplingSampleProcessor(items={"input_position"}, num_points=2)
    value = torch.tensor([1, 2, 3])
    with pytest.raises(NotImplementedError):
        new_key, same_value = precollator.inverse(key="input_position", value=value)
