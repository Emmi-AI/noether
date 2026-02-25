#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from noether.core.providers import PathProvider
from noether.core.schemas.trainers import BaseTrainerConfig
from noether.core.trackers import BaseTracker
from noether.data.container import DataContainer
from noether.training.trainers.base import BaseTrainer
from noether.training.trainers.types import LossResult


class MockTrainer(BaseTrainer):
    def loss_compute(self, forward_output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> LossResult:
        return {"loss": forward_output["loss"]}


@pytest.fixture
def mock_path_provider(tmp_path) -> PathProvider:
    return PathProvider(output_root_path=tmp_path, run_id="test_run", stage_name="test_stage")


@pytest.fixture
def mock_tracker() -> MagicMock:
    return MagicMock(spec=BaseTracker)


@pytest.fixture
def mock_data_container() -> MagicMock:
    data_container = MagicMock(spec=DataContainer)
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 100
    data_container.get_dataset.return_value = mock_dataset
    return data_container


def test_skip_nan_loss_logic(mock_path_provider, mock_tracker, mock_data_container):
    """Integration-style test to verify skip_nan_loss functionality in BaseTrainer.

    Covers:
    - Skipping of NaN losses when enabled.
    - Incrementing and resetting of the skip counter.
    - Raising RuntimeError when max count is exceeded.
    - Recovery from NaN loss when a valid loss occurs.
    """
    config = BaseTrainerConfig(
        kind="mock",
        effective_batch_size=4,
        max_updates=10,
        skip_nan_loss=True,
        skip_nan_loss_max_count=2,
        callbacks=[],
    )

    trainer = MockTrainer(
        config=config,
        data_container=mock_data_container,
        device="cpu",
        tracker=mock_tracker,
        path_provider=mock_path_provider,
    )

    # Mock model for tracking optimizer interactions
    model = MagicMock()
    model.is_frozen = False
    model.optimizer_step = MagicMock()
    model.optimizer_zero_grad = MagicMock()

    nan_loss = torch.tensor(float("nan"))
    valid_loss = torch.tensor(1.0)

    # 1. Encounter NaN loss: should increment counter and skip grad scaler backward
    # We mock grad_scaler since BaseTrainer initializes it
    trainer.grad_scaler = MagicMock()

    trainer._gradient_step(nan_loss, model, accumulation_steps=1, iter_step=0)

    assert trainer.skip_nan_loss_counter == 1
    assert trainer._skip_nan_step is False  # Reset at the end of _gradient_step
    trainer.grad_scaler.scale(nan_loss).backward.assert_not_called()
    model.optimizer_step.assert_not_called()

    # 2. Encounter second NaN loss: should increment counter to 2
    trainer._gradient_step(nan_loss, model, accumulation_steps=1, iter_step=1)

    assert trainer.skip_nan_loss_counter == 2
    trainer.grad_scaler.scale(nan_loss).backward.assert_not_called()
    model.optimizer_step.assert_not_called()

    # 3. Encounter third NaN loss: should exceed skip_nan_loss_max_count=2 and raise error
    with pytest.raises(RuntimeError, match="encountered 2 nan losses in a row"):
        trainer._gradient_step(nan_loss, model, accumulation_steps=1, iter_step=2)

    # 4. Recovery: Reset counter and encounter valid loss
    trainer.skip_nan_loss_counter = 1
    trainer._gradient_step(valid_loss, model, accumulation_steps=1, iter_step=3)

    assert trainer.skip_nan_loss_counter == 0
    # Backward should have been called for valid_loss
    trainer.grad_scaler.scale.assert_called_with(valid_loss)
    # Optimizer step should be called since accumulation_steps=1 and not skipped
    model.optimizer_step.assert_called()


def test_skip_nan_loss_disabled(mock_path_provider, mock_tracker, mock_data_container):
    """Verify that when skip_nan_loss is False, it doesn't skip (or rather, it doesn't do the special handling)."""
    config = BaseTrainerConfig(
        kind="mock",
        effective_batch_size=4,
        max_updates=10,
        skip_nan_loss=False,
        callbacks=[],
    )

    trainer = MockTrainer(
        config=config,
        data_container=mock_data_container,
        device="cpu",
        tracker=mock_tracker,
        path_provider=mock_path_provider,
    )

    model = MagicMock()
    model.is_frozen = False
    model.optimizer_step = MagicMock()

    nan_loss = torch.tensor(float("nan"))
    nan_loss.requires_grad = True  # Ensure it requires grad to test backward call

    # Should call backward even if it's NaN because skipping is disabled
    trainer._gradient_step(nan_loss, model, accumulation_steps=1, iter_step=0)

    assert trainer.skip_nan_loss_counter == 0
    model.optimizer_step.assert_called()


def test_skip_nan_loss_no_nan(mock_path_provider, mock_tracker, mock_data_container):
    config = BaseTrainerConfig(
        kind="mock",
        effective_batch_size=4,
        max_updates=10,
        skip_nan_loss=True,
        callbacks=[],
    )

    trainer = MockTrainer(
        config=config,
        data_container=mock_data_container,
        device="cpu",
        tracker=mock_tracker,
        path_provider=mock_path_provider,
    )

    model = MagicMock()
    model.is_frozen = False
    model.optimizer_step = MagicMock()

    nan_loss = torch.tensor(1.0)
    nan_loss.requires_grad = True  # Ensure it requires grad to test backward call

    # Should call backward even if it's NaN because skipping is disabled
    trainer._gradient_step(nan_loss, model, accumulation_steps=1, iter_step=0)

    assert trainer.skip_nan_loss_counter == 0
    model.optimizer_step.assert_called()
