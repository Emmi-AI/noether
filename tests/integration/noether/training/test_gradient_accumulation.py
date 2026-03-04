#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from noether.core.models import Model
from noether.core.providers import PathProvider
from noether.core.schemas.dataset import DatasetBaseConfig
from noether.core.schemas.models import ModelBaseConfig
from noether.core.schemas.optimizers import OptimizerConfig
from noether.core.schemas.trainers import BaseTrainerConfig
from noether.core.trackers import BaseTracker
from noether.data import Dataset
from noether.data.container import DataContainer
from noether.data.pipeline import MultiStagePipeline
from noether.data.pipeline.collators import DefaultCollator
from noether.training.trainers.base import BaseTrainer
from noether.training.trainers.types import LossResult


class GradAccumTrainer(BaseTrainer):
    """Trainer for gradient accumulation tests.

    Expects dist_model to return {"output": tensor} and computes mean loss over it.
    """

    def loss_compute(
        self,
        forward_output: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> LossResult:
        return {"output_loss": forward_output["output"].mean()}


class SimpleDistModel(nn.Module):
    """Wraps nn.Linear for use as dist_model.

    Accepts keyword argument ``x`` and returns ``{"output": linear(x)}``,
    which is the format expected by GradAccumTrainer.loss_compute.
    """

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"output": self.linear(x)}


class SimpleModelWrapper:
    """Minimal ModelBase-compatible wrapper around a real nn.Module.

    Provides the interface expected by BaseTrainer._gradient_step without
    requiring a full ModelBase implementation.
    """

    def __init__(self, linear: nn.Linear, optimizer: torch.optim.Optimizer) -> None:
        self.linear = linear
        self.optimizer = optimizer
        self.is_frozen = False
        self.nograd_paramnames: list[str] = []

    def optimizer_step(self, grad_scaler) -> None:
        grad_scaler.step(self.optimizer)
        grad_scaler.update()

    def optimizer_zero_grad(self) -> None:
        self.optimizer.zero_grad()


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


def _make_trainer(mock_path_provider, mock_tracker, mock_data_container, **config_kwargs) -> GradAccumTrainer:
    config = BaseTrainerConfig(
        kind="mock",
        effective_batch_size=4,
        max_epochs=10,
        callbacks=[],
        forward_properties=["x"],
        **config_kwargs,
    )
    return GradAccumTrainer(
        config=config,
        data_container=mock_data_container,
        device="cpu",
        tracker=mock_tracker,
        path_provider=mock_path_provider,
    )


def test_update_defers_optimizer_step_during_accumulation(mock_path_provider, mock_tracker, mock_data_container):
    """Verify that update does not call optimizer_step until the final accumulation step."""
    features = 4
    accumulation_steps = 3

    linear = nn.Linear(features, 1, bias=False)
    dist_model = SimpleDistModel(linear)

    model = MagicMock()
    model.is_frozen = False
    model.nograd_paramnames = []

    trainer = _make_trainer(mock_path_provider, mock_tracker, mock_data_container)
    trainer.grad_scaler = MagicMock()
    trainer.grad_scaler.scale.return_value = MagicMock()

    batch = {"x": torch.randn(1, features)}

    for step in range(accumulation_steps - 1):
        trainer.update(
            batch, dist_model, model=model, accumulation_steps_total=accumulation_steps, accumulation_step=step
        )
        model.optimizer_step.assert_not_called()
        model.optimizer_zero_grad.assert_not_called()

    trainer.update(
        batch,
        dist_model,
        model=model,
        accumulation_steps_total=accumulation_steps,
        accumulation_step=accumulation_steps - 1,
    )
    model.optimizer_step.assert_called_once_with(trainer.grad_scaler)
    model.optimizer_zero_grad.assert_called_once()


def test_update_optimizer_step_called_each_full_cycle(mock_path_provider, mock_tracker, mock_data_container):
    """Verify that optimizer_step is called exactly once per full accumulation cycle across multiple cycles."""
    features = 4
    accumulation_steps = 2
    num_cycles = 3

    linear = nn.Linear(features, 1, bias=False)
    dist_model = SimpleDistModel(linear)

    model = MagicMock()
    model.is_frozen = False
    model.nograd_paramnames = []

    trainer = _make_trainer(mock_path_provider, mock_tracker, mock_data_container)
    trainer.grad_scaler = MagicMock()
    trainer.grad_scaler.scale.return_value = MagicMock()

    batch = {"x": torch.randn(1, features)}

    for global_step in range(accumulation_steps * num_cycles):
        trainer.update(
            batch, dist_model, model=model, accumulation_steps_total=accumulation_steps, accumulation_step=global_step
        )

    assert model.optimizer_step.call_count == num_cycles
    assert model.optimizer_zero_grad.call_count == num_cycles


def test_update_loss_divided_by_accumulation_steps(mock_path_provider, mock_tracker, mock_data_container):
    """Verify that update passes loss / accumulation_steps to the backward function.

    The raw loss produced by loss_compute is divided by accumulation_steps inside
    _gradient_step before it reaches backward, so the gradient for each mini-batch
    is proportionally scaled.
    """
    torch.manual_seed(0)
    features = 4
    accumulation_steps = 3

    linear = nn.Linear(features, 1, bias=False)
    dist_model = SimpleDistModel(linear)

    x = torch.randn(1, features)
    batch = {"x": x}

    # Compute the expected raw loss without triggering a backward
    with torch.no_grad():
        expected_raw_loss = dist_model(x=x)["output"].mean().item()

    captured_losses: list[float] = []

    class CapturingScaler:
        class _Proxy:
            def __init__(self, loss: torch.Tensor) -> None:
                self._loss = loss

            def backward(self, **kwargs) -> None:
                self._loss.backward(**kwargs)
                captured_losses.append(self._loss.item())

        def scale(self, loss: torch.Tensor) -> _Proxy:
            return self._Proxy(loss)

        def step(self, optimizer) -> None:
            optimizer.step()

        def update(self) -> None:
            pass

    trainer = _make_trainer(mock_path_provider, mock_tracker, mock_data_container)
    trainer.grad_scaler = CapturingScaler()

    model = MagicMock()
    model.is_frozen = False
    model.nograd_paramnames = []

    trainer.update(batch, dist_model, model=model, accumulation_steps_total=accumulation_steps, accumulation_step=0)

    assert len(captured_losses) == 1
    assert abs(captured_losses[0] - expected_raw_loss / accumulation_steps) < 1e-6


def test_update_accumulated_gradients_equivalent_to_full_batch(mock_path_provider, mock_tracker, mock_data_container):
    """Verify that N accumulation steps via update produce the same parameter update as one step on the full batch.

    For a linear model with mean loss:
        full-batch gradient = (x1 + x2) / 2  (mean over 2 samples)

    With accumulation_steps=2 and per-mini-batch mean loss:
        step 1: gradient += mean_grad(x1) / 2 = x1 / 2
        step 2: gradient += mean_grad(x2) / 2 = x2 / 2
        total:  = (x1 + x2) / 2  ✓

    Both approaches must yield identical parameter values after the optimizer update.
    """
    torch.manual_seed(0)
    features = 4
    lr = 0.1
    accumulation_steps = 2

    x1 = torch.randn(1, features)
    x2 = torch.randn(1, features)
    x_combined = torch.cat([x1, x2])  # shape [2, features]

    # --- Baseline: single update on the full batch ---
    linear_single = nn.Linear(features, 1, bias=False)
    init_weights = linear_single.weight.data.clone()

    dist_model_single = SimpleDistModel(linear_single)
    opt_single = torch.optim.SGD(linear_single.parameters(), lr=lr)
    wrapper_single = SimpleModelWrapper(linear_single, opt_single)
    trainer_single = _make_trainer(mock_path_provider, mock_tracker, mock_data_container)

    trainer_single.update(
        {"x": x_combined}, dist_model_single, model=wrapper_single, accumulation_steps_total=1, accumulation_step=0
    )
    weight_after_single = linear_single.weight.data.clone()

    # --- Accumulated: two updates on mini-batches ---
    linear_accum = nn.Linear(features, 1, bias=False)
    linear_accum.weight.data.copy_(init_weights)  # identical starting point

    dist_model_accum = SimpleDistModel(linear_accum)
    opt_accum = torch.optim.SGD(linear_accum.parameters(), lr=lr)
    wrapper_accum = SimpleModelWrapper(linear_accum, opt_accum)
    trainer_accum = _make_trainer(mock_path_provider, mock_tracker, mock_data_container)

    trainer_accum.update(
        {"x": x1},
        dist_model_accum,
        model=wrapper_accum,
        accumulation_steps_total=accumulation_steps,
        accumulation_step=0,
    )
    trainer_accum.update(
        {"x": x2},
        dist_model_accum,
        model=wrapper_accum,
        accumulation_steps_total=accumulation_steps,
        accumulation_step=1,
    )
    weight_after_accum = linear_accum.weight.data.clone()

    assert torch.allclose(weight_after_single, weight_after_accum, atol=1e-6), (
        f"Single-step weights: {weight_after_single}\n"
        f"Accumulated weights: {weight_after_accum}\n"
        f"Difference: {(weight_after_single - weight_after_accum).abs().max().item()}"
    )


def test_update_gradients_reset_after_full_accumulation_cycle(mock_path_provider, mock_tracker, mock_data_container):
    """Verify that update zeroes gradients after each full accumulation cycle.

    If stale gradients carried over between cycles, subsequent optimizer steps
    would incorporate contributions from previous cycles, corrupting the update.
    """
    torch.manual_seed(0)
    features = 4
    accumulation_steps = 2

    linear = nn.Linear(features, 1, bias=False)
    dist_model = SimpleDistModel(linear)
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)
    wrapper = SimpleModelWrapper(linear, optimizer)
    trainer = _make_trainer(mock_path_provider, mock_tracker, mock_data_container)

    # Complete one full accumulation cycle
    for step in range(accumulation_steps):
        batch = {"x": torch.randn(1, features)}
        trainer.update(
            batch, dist_model, model=wrapper, accumulation_steps_total=accumulation_steps, accumulation_step=step
        )

    # After a full cycle optimizer_zero_grad has been called; the next cycle starts with clean gradients.
    grad = linear.weight.grad
    assert grad is None or torch.all(grad == 0), f"Expected zeroed gradients after full accumulation cycle, got: {grad}"


# ---------------------------------------------------------------------------
# Blackbox test for train()
# ---------------------------------------------------------------------------


class _TinyDataset(Dataset):
    """In-memory dataset of float vectors, compatible with the noether Dataset interface."""

    def __init__(self, num_samples: int, features: int) -> None:
        super().__init__(dataset_config=DatasetBaseConfig(kind="tiny"))
        torch.manual_seed(0)
        self._x = torch.randn(num_samples, features)

    def __len__(self) -> int:
        return len(self._x)

    def getitem_x(self, idx: int) -> torch.Tensor:
        return self._x[idx]


class _TinyLinearModel(Model):
    """Single linear layer model compatible with the noether Model/ModelBase interface."""

    def __init__(self, features: int) -> None:
        super().__init__(
            model_config=ModelBaseConfig(
                kind="test._TinyLinearModel",
                name="tiny_linear",
                optimizer_config=OptimizerConfig(kind="torch.optim.SGD", lr=0.1),
            )
        )
        self.linear = nn.Linear(features, 1, bias=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"output": self.linear(x)}


def test_train_with_gradient_accumulation_updates_weights(mock_path_provider, mock_tracker, tmp_path):
    """Blackbox test: train() with gradient accumulation runs end-to-end and updates model weights.

    Sets up a full training stack — real Dataset, DataContainer, Model, and Trainer — with
    effective_batch_size=4 and max_batch_size=2 so the trainer infers accumulation_steps=2.

    Asserts that model weights change from their initial values, confirming that optimizer steps
    were actually taken through the accumulation loop inside _run_epoch.
    """
    features = 4
    num_samples = 16

    dataset = _TinyDataset(num_samples=num_samples, features=features)
    dataset.pipeline = MultiStagePipeline(collators=[DefaultCollator(items=["x"])])
    data_container = DataContainer(datasets={"train": dataset}, num_workers=0, pin_memory=False)

    model = _TinyLinearModel(features=features)
    initial_weights = model.linear.weight.data.clone()

    config = BaseTrainerConfig(
        kind="test",
        effective_batch_size=4,
        max_epochs=1,
        callbacks=[],
        add_default_callbacks=False,
        add_trainer_callbacks=False,
        forward_properties=["x"],
        disable_gradient_accumulation=False,
        max_batch_size=2,  # → accumulation_steps = effective_batch_size / max_batch_size = 2
    )
    trainer = GradAccumTrainer(
        config=config,
        data_container=data_container,
        device="cpu",
        tracker=mock_tracker,
        path_provider=mock_path_provider,
    )

    trainer.train(model)

    weights_after = model.linear.weight.data
    assert not torch.allclose(initial_weights, weights_after), (
        "Model weights did not change after training with gradient accumulation. "
        "Optimizer steps were not taken, which indicates iter_step is not forwarded "
        "correctly to update() inside _run_epoch."
    )
