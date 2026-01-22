#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

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
