#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from unittest.mock import Mock

import torch
import torch.nn as nn

from noether.core.callbacks import CallbackBase
from noether.core.models.model import Model
from noether.core.providers import PathProvider
from noether.core.schemas import DatasetBaseConfig
from noether.core.schemas.initializers import ResumeInitializerConfig
from noether.core.schemas.models.base import ModelBaseConfig
from noether.core.schemas.optimizers import OptimizerConfig
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


class DummyModelConfig(ModelBaseConfig):
    kind: str = "dummy"
    name: str = "dummy_model"
    optimizer_config: OptimizerConfig = OptimizerConfig(
        kind="torch.optim.AdamW",
        lr=1e-3,
        weight_decay=0.0,
        clip_grad_value=None,
        clip_grad_norm=None,
        weight_decay_schedule=None,
        schedule_config=None,
    )


class DummyModel(Model):
    def __init__(self, config=None, **kwargs):
        model_config = config or DummyModelConfig()
        super().__init__(model_config=model_config, **kwargs)
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
        return torch.tensor(0.0)


class StatefullCallback(CallbackBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0

    def state_dict(self):
        return {"counter": self.counter}

    def load_state_dict(self, state_dict):
        self.counter = state_dict["counter"]

    def resume_from_checkpoint(self, **kwargs):
        pass


def test_trainer_checkpoint_reloading(tmp_path):
    run_id = "test_run"
    output_root = tmp_path / "outputs"
    path_provider = PathProvider(output_root_path=output_root, run_id=run_id, stage_name="train")

    # 1. Setup initial model and trainer
    dataset = DummyDataset(size=100)
    data_container = DataContainer(datasets={"train": dataset})

    model = DummyModel()
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.fill_(2.0)

    config = BaseTrainerConfig(
        kind="base",
        max_epochs=2,
        effective_batch_size=2,
        callbacks=[],
        forward_properties=["x"],
        target_properties=["y"],
    )

    trainer = DummyTrainer(
        config=config,
        data_container=data_container,
        device="cpu",
        tracker=Mock(),
        path_provider=path_provider,
    )

    callback = StatefullCallback(
        trainer=trainer,
        model=model,
        data_container=data_container,
        tracker=trainer.tracker,
        log_writer=trainer.log_writer,
        checkpoint_writer=trainer.checkpoint_writer,
        metric_property_provider=trainer.metric_property_provider,
    )
    callback.counter = 42
    trainer.callbacks = [callback]

    trainer.update_counter.cur_iteration.epoch = 1
    trainer.update_counter.cur_iteration.update = 50
    trainer.update_counter.cur_iteration.sample = 100

    # 2. Save checkpoint
    checkpoint = "E1_U50_S100"

    trainer.checkpoint_writer.save(
        model=model,
        checkpoint_tag=checkpoint,
        trainer=trainer,
        save_weights=True,
        save_optim=False,
    )

    # 3. Setup new model and trainer to reload
    new_model = DummyModel()

    resume_config = ResumeInitializerConfig(
        run_id=run_id,
        stage_name="train",
        model_name=model.name,
        checkpoint=checkpoint,
    )

    new_trainer_config = BaseTrainerConfig(
        kind="base",
        max_epochs=3,
        effective_batch_size=2,
        callbacks=[],
        forward_properties=["x"],
        target_properties=["y"],
        initializer=resume_config,
    )

    new_path_provider = PathProvider(output_root_path=output_root, run_id="new_run", stage_name="train")

    new_trainer = DummyTrainer(
        config=new_trainer_config,
        data_container=data_container,
        device="cpu",
        tracker=Mock(),
        path_provider=new_path_provider,
    )
    new_callback = StatefullCallback(
        trainer=new_trainer,
        model=new_model,
        data_container=data_container,
        tracker=new_trainer.tracker,
        log_writer=new_trainer.log_writer,
        checkpoint_writer=new_trainer.checkpoint_writer,
        metric_property_provider=new_trainer.metric_property_provider,
    )
    assert new_callback.counter == 0
    new_trainer.callbacks = [new_callback]

    # 4. Trigger reload
    new_trainer.apply_resume_initializer(new_model)

    # 5. Assertions
    assert torch.allclose(new_model.linear.weight, model.linear.weight)
    assert torch.allclose(new_model.linear.bias, model.linear.bias)
    assert new_callback.counter == 42
    assert new_trainer.start_checkpoint.epoch == 1
    assert new_trainer.start_checkpoint.update == 50
    assert new_trainer.start_checkpoint.sample == 100
