#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.data.pipeline.sample_processors import PositionNormalizationSampleProcessor


@pytest.fixture
def precollator():
    return PositionNormalizationSampleProcessor(
        items={"position"},
        raw_pos_min=[2.0, 0.0],
        raw_pos_max=[3.0, 2.0],
        scale=1000,
    )


def test_call_preprocess(precollator):
    sample = {
        "position": torch.tensor([[2.0, 1.0], [3.0, 2.0]]),
        "unchanged": torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
    }
    new_sample = precollator(sample)

    assert torch.allclose(new_sample["position"], torch.tensor([[0.0, 500.0], [1000.0, 1000.0]]))
    assert torch.all(new_sample["unchanged"] == sample["unchanged"])


def test_call_preprocess_3d(precollator):
    sample = {
        "position": torch.tensor([[2.0, 1.0], [3.0, 2.0]]).unsqueeze(0),
    }
    new_sample = precollator(sample)

    assert torch.allclose(
        new_sample["position"],
        torch.tensor([[[0.0, 500.0], [1000.0, 1000.0]]]),
    )


def test_call_raises_key_error_for_missing_item(precollator):
    sample = {"unchanged": torch.tensor([[3.0, 4.0], [5.0, 6.0]])}
    with pytest.raises(KeyError):
        precollator(sample)
