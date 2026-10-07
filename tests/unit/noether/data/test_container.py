#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""Tests for the DataContainer DataLoader seeding behavior."""

from unittest.mock import MagicMock, patch

import torch

from noether.core.utils.seed import seed_worker
from noether.data.container import DataContainer


def _make_container(
    seed: int | None,
    num_workers: int = 0,
    multiprocessing_context: str | None = None,
    persistent_workers: bool = False,
) -> DataContainer:
    """Build a DataContainer with a dummy dataset that never gets used (DataLoader is mocked)."""
    dataset = MagicMock()
    # DataContainer refuses empty dataset dicts but otherwise does not touch the dataset here.
    return DataContainer(
        datasets={"train": dataset},
        num_workers=num_workers,
        pin_memory=False,
        seed=seed,
        multiprocessing_context=multiprocessing_context,
        persistent_workers=persistent_workers,
    )


def _call_get_data_loader(container: DataContainer):
    """Drive get_data_loader with an InterleavedSampler that is fully mocked so we can observe the
    kwargs that DataContainer forwards to torch.utils.data.DataLoader."""
    sampler = MagicMock()
    sampler.dataset = MagicMock()
    sampler.batch_sampler = MagicMock()
    sampler.collator = MagicMock()

    with (
        patch("noether.data.container.InterleavedSampler", return_value=sampler),
        patch("noether.data.container.DataLoader") as mock_loader_cls,
    ):
        mock_loader = MagicMock()
        mock_loader.num_workers = container.num_workers
        mock_loader.pin_memory = False
        mock_loader.prefetch_factor = None
        mock_loader_cls.return_value = mock_loader

        container.get_data_loader(
            train_sampler=MagicMock(),
            train_collator=None,
            batch_size=2,
            epochs=1,
            updates=None,
            samples=None,
            callback_samplers=[],
        )

        assert mock_loader_cls.called, "DataLoader should have been constructed"
        return mock_loader_cls.call_args.kwargs


def test_data_container_seeded_passes_generator_and_worker_init_fn():
    seed = 1234
    container = _make_container(seed=seed)
    kwargs = _call_get_data_loader(container)

    assert kwargs["worker_init_fn"] is seed_worker

    generator = kwargs["generator"]
    assert isinstance(generator, torch.Generator)

    # The generator should be seeded deterministically from the given seed: constructing a second
    # generator with the same seed must produce identical draws.
    reference = torch.Generator()
    reference.manual_seed(seed)
    assert torch.equal(
        torch.randint(0, 2**31 - 1, (8,), generator=generator),
        torch.randint(0, 2**31 - 1, (8,), generator=reference),
    )


def test_data_container_unseeded_passes_none():
    container = _make_container(seed=None)
    kwargs = _call_get_data_loader(container)

    assert kwargs["worker_init_fn"] is None
    assert kwargs["generator"] is None


def test_worker_kwargs_forwarded_when_workers_positive():
    container = _make_container(seed=None, num_workers=8, multiprocessing_context="spawn", persistent_workers=True)
    kwargs = _call_get_data_loader(container)

    assert kwargs["num_workers"] == 8
    assert kwargs["multiprocessing_context"] == "spawn"
    assert kwargs["persistent_workers"] is True


def test_worker_kwargs_omitted_when_no_workers():
    # multiprocessing_context / persistent_workers are invalid for num_workers=0; torch raises if
    # they are passed, so DataContainer must drop them entirely rather than forward them.
    container = _make_container(seed=None, num_workers=0, multiprocessing_context="spawn", persistent_workers=True)
    kwargs = _call_get_data_loader(container)

    assert kwargs["num_workers"] == 0
    assert "multiprocessing_context" not in kwargs
    assert "persistent_workers" not in kwargs
