#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""Integration test for BaseTrainer to verify callbacks execute at the last train step."""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from noether.core.callbacks import PeriodicCallback, PeriodicDataIteratorCallback
from noether.core.models.single import Model
from noether.core.schemas import DatasetBaseConfig
from noether.core.schemas.callbacks import CallBackBaseConfig
from noether.core.schemas.trainers import BaseTrainerConfig
from noether.data.base.dataset import Dataset
from noether.data.container import DataContainer
from noether.data.pipeline import Collator
from noether.training.trainers import BaseTrainer


class DummyDataset(Dataset):
    def __init__(self, size=5):
        super().__init__(DatasetBaseConfig(kind="", split="train"))
        self._size, self.pipeline = size, Collator()

    def __len__(self):
        return self._size

    def getitem_x(self, idx):
        return torch.randn(10)

    def getitem_y(self, idx):
        return torch.randint(0, 2, (1,)).squeeze()


class DummyModel(Model):
    def __init__(self, config=None, **kwargs):
        super().__init__(model_config=config or Mock(initializers=[]), **kwargs)
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

    def optimizer_step(self, grad_scaler=None):
        pass

    def optimizer_schedule_step(self):
        pass

    def optimizer_zero_grad(self, set_to_none=True):
        pass


class DummyTrainer(BaseTrainer):
    def loss_compute(self, forward_output, targets):
        return F.cross_entropy(forward_output, targets["y"])


class TrackingCallback(PeriodicCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_calls, self.periodic_calls = [], []

    def track_after_update_step(self, *, update_counter, **_):
        self.update_calls.append(update_counter.is_finished)

    def periodic_callback(self, *, update_counter, **_):
        self.periodic_calls.append(update_counter.is_finished)


class DummyIteratorCallback(PeriodicDataIteratorCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received_batches = []

    def register_sampler_config(self):
        return self._sampler_config_from_key(key="test")

    def process_data(self, batch, *, trainer_model):
        self.received_batches.append(batch)
        return batch

    def process_results(self, results, **_):
        pass


@pytest.mark.filterwarnings("ignore:Batch contains additional keys")
def test_callbacks_execute_at_last_step():
    num_samples, num_epochs = 5, 3
    dataset = DummyDataset(size=num_samples)
    data_container = DataContainer(datasets={"train": dataset})
    trainer = DummyTrainer(
        config=BaseTrainerConfig(
            kind="base",
            max_epochs=num_epochs,
            effective_batch_size=1,
            callbacks=[],
            forward_properties=["x"],
            target_properties=["y"],
        ),
        data_container=data_container,
        device="cpu",
        tracker=None,
        path_provider=None,
    )
    trainer.get_data_loader = Mock(
        return_value=(
            {"x": torch.randn(1, 10), "y": torch.randint(0, 2, (1,))} for _ in range(num_samples * num_epochs)
        )
    )

    model = DummyModel()
    callback = TrackingCallback(
        callback_config=CallBackBaseConfig.model_validate(dict(every_n_epochs=1)),
        trainer=trainer,
        model=model,
        data_container=data_container,
        tracker=Mock(),
        log_writer=Mock(flush=Mock(), finish=Mock()),
        checkpoint_writer=Mock(),
        metric_property_provider=Mock(),
    )
    trainer.get_all_callbacks = lambda _: [callback]

    trainer.train(model)

    assert len(callback.update_calls) == num_samples * num_epochs
    assert len(callback.periodic_calls) == num_epochs
    assert callback.update_calls[-1] is True
    assert callback.periodic_calls[-1] is True


@pytest.mark.filterwarnings("ignore:Batch contains additional keys")
def test_periodic_iterator_callback_receives_all_updates():
    num_samples_train, num_samples_test, num_epochs = 5, 2, 2
    train_dataset = DummyDataset(size=num_samples_train)
    test_dataset = DummyDataset(size=num_samples_test)
    data_container = DataContainer(datasets={"train": train_dataset, "test": test_dataset})

    trainer = DummyTrainer(
        config=BaseTrainerConfig(
            kind="base",
            max_epochs=num_epochs,
            effective_batch_size=1,
            callbacks=[],
            forward_properties=[
                "x",
            ],
            target_properties=["y"],
        ),
        data_container=data_container,
        device="cpu",
        tracker=None,
        path_provider=None,
    )

    # Prepare batches: train batches then test batches for each epoch
    batches = []
    for epoch in range(num_epochs):
        for i in range(num_samples_train):
            batches.append(
                {"x": torch.randn(1, 10), "y": torch.randint(0, 2, (1,)), "id": torch.tensor([epoch * 100 + i])}
            )
        for i in range(num_samples_test):
            batches.append(
                {"x": torch.randn(1, 10), "y": torch.randint(0, 2, (1,)), "id": torch.tensor([(epoch + 10) * 100 + i])}
            )

    trainer.get_data_loader = Mock(return_value=iter(batches))

    model = DummyModel()
    callback = DummyIteratorCallback(
        callback_config=CallBackBaseConfig.model_validate(dict(every_n_epochs=1)),
        trainer=trainer,
        model=model,
        data_container=data_container,
        tracker=Mock(),
        log_writer=Mock(flush=Mock(), finish=Mock()),
        checkpoint_writer=Mock(),
        metric_property_provider=Mock(),
    )
    trainer.get_all_callbacks = lambda _: [callback]
    callback._sampler_config = callback.register_sampler_config()

    trainer.train(model)

    # Check that we received all test batches
    assert len(callback.received_batches) == num_samples_test * num_epochs
    for epoch in range(num_epochs):
        for i in range(num_samples_test):
            batch = callback.received_batches[epoch * num_samples_test + i]
            assert batch["id"].item() == (epoch + 10) * 100 + i


@pytest.mark.filterwarnings("ignore:Batch contains additional keys")
def test_periodic_iterator_callback_with_gradient_accumulation():
    # effective_batch_size=2, max_batch_size=1 -> accumulation_steps=2
    num_samples_train, num_samples_test, num_epochs = 4, 2, 1
    train_dataset = DummyDataset(size=num_samples_train)
    test_dataset = DummyDataset(size=num_samples_test)
    data_container = DataContainer(datasets={"train": train_dataset, "test": test_dataset})

    trainer = DummyTrainer(
        config=BaseTrainerConfig(
            kind="base",
            max_epochs=num_epochs,
            effective_batch_size=2,
            max_batch_size=1,
            disable_gradient_accumulation=False,
            callbacks=[],
            forward_properties=["x"],
            target_properties=["y"],
        ),
        data_container=data_container,
        device="cpu",
        tracker=None,
        path_provider=None,
    )

    # Prepare batches: 4 train batches, then 2 test batches
    batches = []
    for i in range(num_samples_train):
        batches.append({"x": torch.randn(1, 10), "y": torch.randint(0, 2, (1,)), "id": torch.tensor([i])})
    for i in range(num_samples_test):
        batches.append({"x": torch.randn(1, 10), "y": torch.randint(0, 2, (1,)), "id": torch.tensor([100 + i])})

    trainer.get_data_loader = Mock(return_value=iter(batches))

    model = DummyModel()
    callback = DummyIteratorCallback(
        callback_config=CallBackBaseConfig.model_validate(dict(every_n_epochs=1)),
        trainer=trainer,
        model=model,
        data_container=data_container,
        tracker=Mock(),
        log_writer=Mock(flush=Mock(), finish=Mock()),
        checkpoint_writer=Mock(),
        metric_property_provider=Mock(),
    )
    trainer.get_all_callbacks = lambda _: [callback]
    callback._sampler_config = callback.register_sampler_config()

    trainer.train(model)

    # Check that we received all test batches
    assert len(callback.received_batches) == num_samples_test
    for i in range(num_samples_test):
        assert callback.received_batches[i]["id"].item() == 100 + i
