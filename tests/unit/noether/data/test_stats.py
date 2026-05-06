#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.data.stats import RunningMoments


def test_push_scalar():
    data = torch.rand(100, generator=torch.Generator().manual_seed(0))
    expected_mean = data.double().mean()
    expected_std = data.double().std()
    stats = RunningMoments()
    for item in data:
        stats.push_scalar(item.item())
    assert torch.allclose(stats.mean, expected_mean, atol=1e-5)
    assert torch.allclose(stats.std, expected_std, atol=1e-5)


def test_push_tensor_fullbatch_2d():
    data = torch.rand(100, 3, generator=torch.Generator().manual_seed(0))
    expected_mean = data.double().mean(dim=0)
    expected_std = data.double().std(dim=0)
    stats = RunningMoments()
    stats.push_tensor(data)
    assert torch.allclose(expected_mean, stats.mean, atol=1e-5)
    assert torch.allclose(expected_std, stats.std, atol=1e-5)


def test_push_tensor_minibatch_2d():
    data = torch.rand(100, 3, generator=torch.Generator().manual_seed(0))
    expected_mean = data.double().mean(dim=0)
    expected_std = data.double().std(dim=0)
    stats = RunningMoments()
    for chunk in data.chunk(4):
        stats.push_tensor(chunk)
    assert torch.allclose(expected_mean, stats.mean, atol=1e-5)
    assert torch.allclose(expected_std, stats.std, atol=1e-5)


def test_push_tensor_minibatch_3d_dim1():
    data = torch.rand(100, 3, 4, generator=torch.Generator().manual_seed(0))
    expected_mean = data.double().mean(dim=[0, 2])
    expected_std = data.double().std(dim=[0, 2])
    stats = RunningMoments()
    for chunk in data.chunk(4):
        stats.push_tensor(chunk, dim=1)
    assert torch.allclose(expected_mean, stats.mean, atol=1e-5)
    assert torch.allclose(expected_std, stats.std, atol=1e-5)


def test_push_tensor_minibatch_3d_dim2():
    data = torch.rand(100, 3, 4, generator=torch.Generator().manual_seed(0))
    expected_mean = data.double().mean(dim=[0, 1])
    expected_std = data.double().std(dim=[0, 1])
    stats = RunningMoments()
    for chunk in data.chunk(4):
        stats.push_tensor(chunk, dim=2)
    assert torch.allclose(expected_mean, stats.mean, atol=1e-5)
    assert torch.allclose(expected_std, stats.std, atol=1e-5)


def test_merge_inplace_two_nonempty():
    """Combined moments must equal a single-pass over the concatenated data."""
    g = torch.Generator().manual_seed(0)
    data_a = torch.rand(40, 3, generator=g)
    data_b = torch.rand(60, 3, generator=g)
    full = torch.cat([data_a, data_b], dim=0)

    full_stats = RunningMoments()
    full_stats.push_tensor(full)

    a = RunningMoments()
    a.push_tensor(data_a)
    b = RunningMoments()
    b.push_tensor(data_b)
    a.merge_(b)

    assert a.count == full_stats.count
    assert torch.allclose(a.mean, full_stats.mean, atol=1e-12)
    assert torch.allclose(a.std, full_stats.std, atol=1e-12)
    assert torch.allclose(a._min, full_stats._min, atol=1e-12)
    assert torch.allclose(a._max, full_stats._max, atol=1e-12)


def test_merge_inplace_log_scale_consistent():
    """log_scale flag must match between merger and mergee."""
    a = RunningMoments(log_scale=False)
    a.push_tensor(torch.rand(10, 2))
    b = RunningMoments(log_scale=True)
    b.push_tensor(torch.rand(10, 2))
    with pytest.raises(ValueError, match="log_scale"):
        a.merge_(b)
