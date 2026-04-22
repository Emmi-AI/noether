#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""Tests for the DataContainer DataLoader seeding behavior."""

from unittest.mock import MagicMock, patch

import torch

from noether.core.utils.seed import seed_worker
from noether.data.container import DataContainer


def _make_container(seed: int | None) -> DataContainer:
    """Build a DataContainer with a dummy dataset that never gets used (DataLoader is mocked)."""
    dataset = MagicMock()
    # DataContainer refuses empty dataset dicts but otherwise does not touch the dataset here.
    return DataContainer(datasets={"train": dataset}, num_workers=0, pin_memory=False, seed=seed)


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
        mock_loader.num_workers = 0
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
