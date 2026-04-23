#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Integration test: a MultiStagePipeline whose sample processors use the global torch RNG must
produce the *exact same* batches across runs when the training seed is set and the DataLoader is
configured with a seeded generator + ``seed_worker`` ``worker_init_fn``.

This is the end-to-end guarantee behind the bug where ABUPT training on ``aero_cfd`` showed variance
between runs that used the same seed: without worker seeding the ``torch.randperm`` inside
``PointSamplingSampleProcessor`` / ``SupernodeSamplingSampleProcessor`` was driven by a per-worker
RNG that drifted between runs.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from noether.core.utils.seed import seed_worker, set_seed
from noether.data.pipeline.multistage import MultiStagePipeline
from noether.data.pipeline.sample_processors.point_sampling import PointSamplingSampleProcessor
from noether.data.pipeline.sample_processors.supernode_sampling import SupernodeSamplingSampleProcessor


class _FixedPointCloudDataset(Dataset):
    """Tiny dataset of fixed pointclouds with no internal randomness.

    Each sample is generated once at construction time from a dedicated generator so the dataset
    itself contributes zero variance — any difference observed between runs is purely from the
    pipeline / DataLoader RNG path.
    """

    def __init__(self, num_samples: int, points_per_sample: int, feature_dim: int = 3):
        gen = torch.Generator().manual_seed(0)
        self._items: list[dict[str, torch.Tensor | int]] = [
            {
                "input_position": torch.randn(points_per_sample, feature_dim, generator=gen),
                "index": i,
            }
            for i in range(num_samples)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        # Return a shallow copy: MultiStagePipeline already deep-copies each sample before
        # handing it to the sample processors, but keeping this dataset side-effect-free avoids
        # worker-to-worker coupling through shared tensors.
        return dict(self._items[idx])


def _build_pipeline() -> MultiStagePipeline:
    """A pipeline with two stochastic sample processors that both rely on the global torch RNG."""
    return MultiStagePipeline(
        sample_processors=[
            PointSamplingSampleProcessor(items={"input_position"}, num_points=32),
            SupernodeSamplingSampleProcessor(
                item="input_position",
                num_supernodes=4,
                items_at_supernodes={"input_position"},
            ),
        ],
    )


def _drain(loader: DataLoader) -> list[dict[str, torch.Tensor]]:
    """Materialize every batch so we can compare across runs without lazy-iterator surprises."""
    return [{k: v.clone() for k, v in batch.items() if torch.is_tensor(v)} for batch in loader]


def _assert_batches_equal(run_a: list[dict[str, torch.Tensor]], run_b: list[dict[str, torch.Tensor]]) -> None:
    assert len(run_a) == len(run_b), f"different number of batches: {len(run_a)} vs {len(run_b)}"
    for i, (ba, bb) in enumerate(zip(run_a, run_b, strict=True)):
        assert ba.keys() == bb.keys(), f"batch {i} key mismatch: {ba.keys()} vs {bb.keys()}"
        for k in ba:
            assert torch.equal(ba[k], bb[k]), f"batch {i} key {k!r} differs"


def _run_pipeline(seed: int, num_workers: int) -> list[dict[str, torch.Tensor]]:
    set_seed(seed)
    dataset = _FixedPointCloudDataset(num_samples=8, points_per_sample=64)
    pipeline = _build_pipeline()

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=pipeline,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )
    return _drain(loader)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_multistage_pipeline_with_sampling_processors_is_deterministic(num_workers: int):
    """Two runs with the same seed must produce identical batches — both when sampling happens
    in the main process (``num_workers=0``) and when it happens in forked workers
    (``num_workers>0``, which is where the worker seeding bug used to bite)."""
    run_a = _run_pipeline(seed=1337, num_workers=num_workers)
    run_b = _run_pipeline(seed=1337, num_workers=num_workers)
    _assert_batches_equal(run_a, run_b)


def test_multistage_pipeline_with_sampling_processors_changes_with_seed():
    """Sanity check: different seeds must produce different batches, otherwise the test above
    would trivially pass even if seeding was broken."""
    run_a = _run_pipeline(seed=1337, num_workers=0)
    run_b = _run_pipeline(seed=7331, num_workers=0)

    # At least one tensor across the run must differ — the sampling is ordered + subset selection,
    # so identical output would be extremely unlikely.
    any_difference = any(not torch.equal(a[k], b[k]) for a, b in zip(run_a, run_b, strict=True) for k in a if k in b)
    assert any_difference, "different seeds produced identical batches — seeding is not taking effect"
