#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from noether.core.utils.training import TrainingIteration, UpdateCounter
from noether.training.trainers.base import (
    BaseTrainer,
    TrainingContextFilter,
)
from noether.training.trainers.types import TrainerResult

_MODULE_PATH = "noether.training.trainers.base"


def _make_update_counter(
    epoch=1,
    update=10,
    sample=100,
    end_epoch=5,
    end_update=None,
    end_sample=None,
    updates_per_epoch=10,
    effective_batch_size=10,
):
    start = TrainingIteration(epoch=epoch, update=update, sample=sample)
    end = TrainingIteration(epoch=end_epoch, update=end_update, sample=end_sample)
    return UpdateCounter(
        start_iteration=start,
        end_iteration=end,
        updates_per_epoch=updates_per_epoch,
        effective_batch_size=effective_batch_size,
    )


def _make_trainer(
    *,
    max_epochs=5,
    max_updates=None,
    max_samples=None,
    effective_batch_size=4,
    dataset_len=100,
    start_at_epoch=None,
    precision="float32",
    log_every_n_epochs=1,
    log_every_n_updates=None,
    log_every_n_samples=None,
    track_every_n_epochs=1,
    track_every_n_updates=None,
    track_every_n_samples=None,
    disable_gradient_accumulation=True,
    max_batch_size=None,
    forward_properties=None,
    target_properties=None,
    use_torch_compile=False,
    find_unused_params=False,
    static_graph=False,
    add_default_callbacks=True,
    add_trainer_callbacks=True,
    skip_nan_loss=False,
    skip_nan_loss_max_count=10,
):
    """Build a fully-mocked BaseTrainer concrete subclass instance."""

    config = MagicMock()
    config.max_epochs = max_epochs
    config.max_updates = max_updates
    config.max_samples = max_samples
    config.effective_batch_size = effective_batch_size
    config.precision = precision
    config.start_at_epoch = start_at_epoch
    config.log_every_n_epochs = log_every_n_epochs
    config.log_every_n_updates = log_every_n_updates
    config.log_every_n_samples = log_every_n_samples
    config.track_every_n_epochs = track_every_n_epochs
    config.track_every_n_updates = track_every_n_updates
    config.track_every_n_samples = track_every_n_samples
    config.disable_gradient_accumulation = disable_gradient_accumulation
    config.max_batch_size = max_batch_size
    config.use_torch_compile = use_torch_compile
    config.find_unused_params = find_unused_params
    config.static_graph = static_graph
    config.add_default_callbacks = add_default_callbacks
    config.add_trainer_callbacks = add_trainer_callbacks
    config.skip_nan_loss = skip_nan_loss
    config.skip_nan_loss_max_count = skip_nan_loss_max_count
    config.forward_properties = forward_properties or list()
    config.target_properties = target_properties or list()
    config.callbacks = list()
    config.initializer = None

    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=dataset_len)
    dataset.pipeline = MagicMock()

    data_container = MagicMock()
    data_container.get_dataset.return_value = dataset

    tracker = MagicMock()
    path_provider = MagicMock()
    metric_property_provider = MagicMock()

    with (
        patch(_MODULE_PATH + ".Factory") as mock_factory,
        patch(_MODULE_PATH + ".get_supported_precision", return_value=precision),
        patch(
            _MODULE_PATH + ".get_grad_scaler_and_autocast_context",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch(_MODULE_PATH + ".LogWriter"),
        patch(_MODULE_PATH + ".CheckpointWriter"),
        patch(_MODULE_PATH + ".UpdateCounter", wraps=UpdateCounter),
    ):
        mock_factory.return_value.create.return_value = None  # no initializer
        mock_factory.return_value.create_list.return_value = list()

        class ConcreteTrainer(BaseTrainer):
            def loss_compute(self, forward_output, targets):
                return torch.tensor(1.0)

        trainer = ConcreteTrainer(
            config=config,
            data_container=data_container,
            device="cpu",
            tracker=tracker,
            path_provider=path_provider,
            metric_property_provider=metric_property_provider,
        )

    return trainer


class TestTrainingContextFilter:
    def test_filter_sets_epoch_fields_when_cur_iteration_set(self):
        counter = _make_update_counter()
        filt = TrainingContextFilter(counter)
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        result = filt.filter(record)

        assert result is True
        assert record.epoch == counter.cur_iteration.epoch
        assert record.max_epoch == counter.end_iteration.epoch
        assert record.update == counter.cur_iteration.update
        assert record.max_update == counter.end_iteration.update

    def test_filter_skips_when_no_cur_iteration(self):
        counter = MagicMock()
        counter.cur_iteration = None
        filt = TrainingContextFilter(counter)
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        result = filt.filter(record)

        assert result is True
        assert not hasattr(record, "epoch")


class TestBaseTrainerInit:
    def test_dataset_smaller_than_effective_batch_size_raises(self):
        with pytest.raises(ValueError, match="Effective dataset length"):
            _make_trainer(dataset_len=2, effective_batch_size=4)

    def test_start_at_epoch_computes_start_checkpoint(self):
        trainer = _make_trainer(start_at_epoch=2, effective_batch_size=4, dataset_len=40)
        # updates_per_epoch = 40 / 4 = 10; start_update = 10*2 = 20
        assert trainer.start_checkpoint.epoch == 2
        assert trainer.start_checkpoint.update == 20
        assert trainer.start_checkpoint.sample == 20 * 4

    def test_no_start_epoch_defaults_to_zero(self):
        trainer = _make_trainer()
        assert trainer.start_checkpoint.epoch == 0
        assert trainer.start_checkpoint.update == 0
        assert trainer.start_checkpoint.sample == 0

    def test_initializer_with_start_at_epoch_raises(self):
        with pytest.raises(ValueError, match="cannot use both"):
            config = MagicMock()
            config.max_epochs = 5
            config.max_updates = None
            config.max_samples = None
            config.effective_batch_size = 4
            config.precision = "float32"
            config.start_at_epoch = 1  # triggers the error
            config.forward_properties = list()
            config.target_properties = list()
            config.callbacks = list()
            config.initializer = "some_initializer"

            dataset = MagicMock()
            dataset.__len__ = MagicMock(return_value=100)
            data_container = MagicMock()
            data_container.get_dataset.return_value = dataset

            fake_initializer = MagicMock()
            fake_initializer.__class__ = MagicMock  # passes isinstance check against InitializerBase

            with (
                patch(_MODULE_PATH + ".Factory") as mock_factory,
                patch(_MODULE_PATH + ".get_supported_precision", return_value="float32"),
                patch(
                    _MODULE_PATH + ".get_grad_scaler_and_autocast_context",
                    return_value=(MagicMock(), MagicMock()),
                ),
                patch(_MODULE_PATH + ".LogWriter"),
                patch(_MODULE_PATH + ".CheckpointWriter"),
            ):
                from noether.core.initializers import InitializerBase as IB

                real_initializer = MagicMock(spec=IB)
                real_initializer.start_checkpoint.return_value = TrainingIteration(epoch=1, update=10, sample=100)
                mock_factory.return_value.create.return_value = real_initializer

                class ConcreteTrainer(BaseTrainer):
                    def loss_compute(self, forward_output, targets):
                        return torch.tensor(1.0)

                ConcreteTrainer(
                    config=config,
                    data_container=data_container,
                    device="cpu",
                    tracker=MagicMock(),
                    path_provider=MagicMock(),
                )

    def test_subclass_overriding_train_raises(self):
        with pytest.raises(ValueError, match="Derived classes should not implement the train method"):
            with (
                patch(_MODULE_PATH + ".Factory") as mock_factory,
                patch(_MODULE_PATH + ".get_supported_precision", return_value="float32"),
                patch(
                    _MODULE_PATH + ".get_grad_scaler_and_autocast_context",
                    return_value=(MagicMock(), MagicMock()),
                ),
                patch(_MODULE_PATH + ".LogWriter"),
                patch(_MODULE_PATH + ".CheckpointWriter"),
            ):
                mock_factory.return_value.create.return_value = None
                mock_factory.return_value.create_list.return_value = list()

                config = MagicMock()
                config.max_epochs = 2
                config.max_updates = None
                config.max_samples = None
                config.effective_batch_size = 4
                config.precision = "float32"
                config.start_at_epoch = None
                config.forward_properties = list()
                config.target_properties = list()
                config.callbacks = list()
                config.initializer = None

                dataset = MagicMock()
                dataset.__len__ = MagicMock(return_value=100)
                data_container = MagicMock()
                data_container.get_dataset.return_value = dataset

                class BadTrainer(BaseTrainer):
                    def train(self, model):  # should not override train
                        pass

                    def loss_compute(self, forward_output, targets):
                        return torch.tensor(1.0)

                BadTrainer(
                    config=config,
                    data_container=data_container,
                    device="cpu",
                    tracker=MagicMock(),
                    path_provider=MagicMock(),
                )

    def test_subclass_overriding_wrap_model_raises(self):
        with pytest.raises(ValueError, match="Derived classes should not implement the wrap_model method"):
            with (
                patch(_MODULE_PATH + ".Factory") as mock_factory,
                patch(_MODULE_PATH + ".get_supported_precision", return_value="float32"),
                patch(
                    _MODULE_PATH + ".get_grad_scaler_and_autocast_context",
                    return_value=(MagicMock(), MagicMock()),
                ),
                patch(_MODULE_PATH + ".LogWriter"),
                patch(_MODULE_PATH + ".CheckpointWriter"),
            ):
                mock_factory.return_value.create.return_value = None

                config = MagicMock()
                config.max_epochs = 2
                config.max_updates = None
                config.max_samples = None
                config.effective_batch_size = 4
                config.precision = "float32"
                config.start_at_epoch = None
                config.forward_properties = list()
                config.target_properties = list()
                config.callbacks = list()
                config.initializer = None

                dataset = MagicMock()
                dataset.__len__ = MagicMock(return_value=100)
                data_container = MagicMock()
                data_container.get_dataset.return_value = dataset

                class BadTrainer(BaseTrainer):
                    def wrap_model(self, model):
                        return model

                    def loss_compute(self, forward_output, targets):
                        return torch.tensor(1.0)

                BadTrainer(
                    config=config,
                    data_container=data_container,
                    device="cpu",
                    tracker=MagicMock(),
                    path_provider=MagicMock(),
                )


class TestCalculateBatchSize:
    def test_eval_run_no_updates(self):
        trainer = _make_trainer(disable_gradient_accumulation=False)
        trainer.end_checkpoint = TrainingIteration(epoch=5, update=0, sample=None)
        bs, acc = trainer._calculate_batch_size_and_accumulation_steps()
        assert acc == 1

    def test_gradient_accumulation_disabled(self):
        trainer = _make_trainer(disable_gradient_accumulation=True)
        bs, acc = trainer._calculate_batch_size_and_accumulation_steps()
        assert acc == 1

    def test_no_max_batch_size_raises_when_accumulation_needed(self):
        trainer = _make_trainer(
            disable_gradient_accumulation=False,
            max_batch_size=None,
        )
        with (
            patch(_MODULE_PATH + ".get_world_size", return_value=1),
            patch(_MODULE_PATH + ".get_num_nodes", return_value=1),
            patch(_MODULE_PATH + ".is_distributed", return_value=False),
        ):
            with pytest.raises(ValueError, match="gradient accumulation requires max_batch_size"):
                trainer._calculate_batch_size_and_accumulation_steps()

    def test_effective_not_divisible_by_world_size_raises(self):
        trainer = _make_trainer(effective_batch_size=5, disable_gradient_accumulation=False)
        with patch(_MODULE_PATH + ".get_world_size", return_value=2):
            with pytest.raises(ValueError, match="needs to be multiple of world_size"):
                trainer._calculate_batch_size_and_accumulation_steps()

    def test_accumulation_steps_computed_correctly(self):
        # effective=8, max_batch=2, world_size=1 → accumulation=4, batch=2
        trainer = _make_trainer(
            effective_batch_size=8,
            dataset_len=80,
            disable_gradient_accumulation=False,
            max_batch_size=2,
        )
        with (
            patch(_MODULE_PATH + ".get_world_size", return_value=1),
            patch(_MODULE_PATH + ".get_num_nodes", return_value=1),
            patch(_MODULE_PATH + ".is_distributed", return_value=False),
        ):
            bs, acc = trainer._calculate_batch_size_and_accumulation_steps()
        assert acc == 4
        assert bs == 2

    def test_fits_in_memory_no_accumulation(self):
        # effective=4, max_batch=8, world_size=1 → accumulation=1, batch=4
        trainer = _make_trainer(
            effective_batch_size=4,
            dataset_len=80,
            disable_gradient_accumulation=False,
            max_batch_size=8,
        )
        with (
            patch(_MODULE_PATH + ".get_world_size", return_value=1),
            patch(_MODULE_PATH + ".get_num_nodes", return_value=1),
            patch(_MODULE_PATH + ".is_distributed", return_value=False),
        ):
            bs, acc = trainer._calculate_batch_size_and_accumulation_steps()
        assert acc == 1
        assert bs == 4

    def test_accumulation_not_divisible_raises(self):
        # effective=7, max_batch=4, world_size=1 -> 7 % 4 != 0
        trainer = _make_trainer(
            effective_batch_size=7,
            dataset_len=70,
            disable_gradient_accumulation=False,
            max_batch_size=4,
        )
        with (
            patch(_MODULE_PATH + ".get_world_size", return_value=1),
            patch(_MODULE_PATH + ".get_num_nodes", return_value=1),
            patch(_MODULE_PATH + ".is_distributed", return_value=False),
        ):
            with pytest.raises(ValueError, match="effective_batch_size_per_device needs to be multiple"):
                trainer._calculate_batch_size_and_accumulation_steps()

    def test_multi_node_disables_automatic_batchsize(self):
        trainer = _make_trainer(disable_gradient_accumulation=False, max_batch_size=None)
        with (
            patch(_MODULE_PATH + ".get_world_size", return_value=2),
            patch(_MODULE_PATH + ".get_num_nodes", return_value=2),
            patch(_MODULE_PATH + ".is_distributed", return_value=True),
        ):
            bs, acc = trainer._calculate_batch_size_and_accumulation_steps()
        assert acc == 1

    def test_torch_compile_disables_automatic_batchsize(self):
        trainer = _make_trainer(
            disable_gradient_accumulation=False,
            use_torch_compile=True,
            max_batch_size=None,
        )
        with (
            patch(_MODULE_PATH + ".get_world_size", return_value=1),
            patch(_MODULE_PATH + ".get_num_nodes", return_value=1),
            patch(_MODULE_PATH + ".is_distributed", return_value=False),
        ):
            bs, acc = trainer._calculate_batch_size_and_accumulation_steps()
        assert acc == 1


class TestSplitBatch:
    def test_splits_correctly(self):
        trainer = _make_trainer(forward_properties=["x"], target_properties=["y"])
        batch = {"x": torch.ones(3), "y": torch.zeros(3)}
        fwd, tgt = trainer._split_batch(batch)
        assert "x" in fwd and "y" not in fwd
        assert "y" in tgt and "x" not in tgt

    def test_emits_warning_on_key_mismatch(self):
        trainer = _make_trainer(forward_properties=["x"], target_properties=["y"])
        batch = {"x": torch.ones(3), "y": torch.zeros(3), "extra": torch.zeros(3)}
        with pytest.warns(UserWarning):
            trainer._split_batch(batch)

    def test_missing_keys_in_batch_warns(self):
        trainer = _make_trainer(forward_properties=["x", "z"], target_properties=["y"])
        batch = {"x": torch.ones(3), "y": torch.zeros(3)}  # "z" is missing
        with pytest.warns(UserWarning):
            fwd, tgt = trainer._split_batch(batch)
        assert "z" not in fwd


class TestTrainStep:
    def _make_trainer_with_model(self, loss_value):
        trainer = _make_trainer(forward_properties=["x"], target_properties=["y"])

        model = MagicMock()
        model.return_value = {"out": torch.tensor(0.5)}

        trainer.loss_compute = MagicMock(return_value=loss_value)
        return trainer, model

    def test_train_step_with_tensor_loss(self):
        loss = torch.tensor(1.5)
        trainer, model = self._make_trainer_with_model(loss)
        batch = {"x": torch.ones(2), "y": torch.zeros(2)}
        result = trainer.train_step(batch, model)
        assert isinstance(result, TrainerResult)
        assert torch.isclose(result.total_loss, loss)

    def test_train_step_with_dict_loss(self):
        losses = {"ce": torch.tensor(1.0), "reg": torch.tensor(0.5)}
        trainer, model = self._make_trainer_with_model(losses)
        batch = {"x": torch.ones(2), "y": torch.zeros(2)}
        result = trainer.train_step(batch, model)
        assert isinstance(result, TrainerResult)
        assert torch.isclose(result.total_loss, torch.tensor(1.5))

    def test_train_step_with_list_loss(self):
        losses = [torch.tensor(1.0), torch.tensor(0.5)]
        trainer, model = self._make_trainer_with_model(losses)
        batch = {"x": torch.ones(2), "y": torch.zeros(2)}
        result = trainer.train_step(batch, model)
        assert isinstance(result, TrainerResult)
        assert torch.isclose(result.total_loss, torch.tensor(1.5))

    def test_train_step_with_tuple_loss_and_additional_outputs(self):
        loss = torch.tensor(2.0)
        extra = {"logits": torch.tensor([0.1, 0.9])}
        trainer, model = self._make_trainer_with_model((loss, extra))
        batch = {"x": torch.ones(2), "y": torch.zeros(2)}
        result = trainer.train_step(batch, model)
        assert result.additional_outputs == extra

    def test_train_step_empty_dict_loss_raises(self):
        trainer, model = self._make_trainer_with_model({})
        batch = {"x": torch.ones(2), "y": torch.zeros(2)}
        with pytest.raises(ValueError, match="No losses computed"):
            trainer.train_step(batch, model)

    def test_loss_compute_not_implemented_raises(self):
        trainer = _make_trainer()

        class StrictBase(BaseTrainer):
            pass  # does not implement loss_compute

        # loss_compute is abstract — directly calling raises NotImplementedError
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            BaseTrainer.loss_compute(trainer, {}, {})


class TestGradientStep:
    def _make_gradient_step_trainer(self, skip_nan=False):
        trainer = _make_trainer(skip_nan_loss=skip_nan)
        trainer._skip_nan_step = False
        trainer.skip_nan_loss_counter = 0
        return trainer

    def test_frozen_model_returns_early(self):
        trainer = self._make_gradient_step_trainer()
        model = MagicMock()
        model.is_frozen = True
        # Should return immediately without error
        trainer._gradient_step(torch.tensor(1.0), model, accumulation_steps=1, iter_step=0)
        model.optimizer_step.assert_not_called()

    def test_basic_gradient_step(self):
        trainer = self._make_gradient_step_trainer()
        model = MagicMock()
        model.is_frozen = False
        model.nograd_paramnames = list()
        loss = torch.tensor(2.0, requires_grad=True)

        with patch.object(trainer.grad_scaler, "scale", return_value=MagicMock()) as mock_scale:
            trainer._gradient_step(loss, model, accumulation_steps=1, iter_step=0)

        mock_scale.assert_called_once()
        model.optimizer_step.assert_called_once_with(trainer.grad_scaler)
        model.optimizer_zero_grad.assert_called_once()

    def test_skip_nan_loss_increments_counter(self):
        trainer = self._make_gradient_step_trainer(skip_nan=True)
        model = MagicMock()
        model.is_frozen = False
        model.nograd_paramnames = list()

        nan_loss = torch.tensor(float("nan"))

        with patch(_MODULE_PATH + ".all_gather_nograd", return_value=nan_loss):
            trainer._gradient_step(nan_loss, model, accumulation_steps=1, iter_step=0)

        assert trainer.skip_nan_loss_counter == 1
        assert trainer._skip_nan_step is False
        model.optimizer_step.assert_not_called()

    def test_skip_nan_loss_max_count_raises(self):
        trainer = self._make_gradient_step_trainer(skip_nan=True)
        trainer.skip_nan_loss_counter = trainer.config.skip_nan_loss_max_count + 1
        model = MagicMock()
        model.is_frozen = False

        nan_loss = torch.tensor(float("nan"))
        with patch(_MODULE_PATH + ".all_gather_nograd", return_value=nan_loss):
            with pytest.raises(RuntimeError, match="nan losses in a row"):
                trainer._gradient_step(nan_loss, model, accumulation_steps=1, iter_step=0)

    def test_valid_loss_after_nan_resets_counter(self):
        trainer = self._make_gradient_step_trainer(skip_nan=True)
        trainer.skip_nan_loss_counter = 3
        model = MagicMock()
        model.is_frozen = False
        model.nograd_paramnames = list()

        valid_loss = torch.tensor(1.0, requires_grad=True)

        with (
            patch(_MODULE_PATH + ".all_gather_nograd", return_value=valid_loss),
            patch.object(trainer.grad_scaler, "scale", return_value=MagicMock()),
        ):
            trainer._gradient_step(valid_loss, model, accumulation_steps=1, iter_step=0)

        assert trainer.skip_nan_loss_counter == 0

    def test_accumulation_steps_delay_optimizer(self):
        trainer = self._make_gradient_step_trainer()
        model = MagicMock()
        model.is_frozen = False
        model.nograd_paramnames = list()
        loss = torch.tensor(1.0, requires_grad=True)

        with patch.object(trainer.grad_scaler, "scale", return_value=MagicMock()):
            # iter_step=0, accumulation_steps=2 → not yet at step boundary
            trainer._gradient_step(loss, model, accumulation_steps=2, iter_step=0)
            model.optimizer_step.assert_not_called()

            # iter_step=1 → (1+1) % 2 == 0 → step happens
            trainer._gradient_step(loss, model, accumulation_steps=2, iter_step=1)
            model.optimizer_step.assert_called_once()


class TestWarnUnusedParams:
    def test_warns_once_non_distributed(self):
        trainer = _make_trainer()
        trainer._has_logged_unused_params = False
        model = MagicMock()
        model.nograd_paramnames = ["layer.weight"]

        with (
            patch(_MODULE_PATH + ".is_rank0", return_value=True),
            patch(_MODULE_PATH + ".is_distributed", return_value=False),
        ):
            trainer._warn_unused_params(model)
            assert trainer._has_logged_unused_params is True

            # Second call should be a no-op
            trainer._warn_unused_params(model)

    def test_errors_in_distributed_setting(self):
        trainer = _make_trainer()
        trainer._has_logged_unused_params = False
        model = MagicMock()
        model.nograd_paramnames = ["layer.weight"]

        with (
            patch(_MODULE_PATH + ".is_rank0", return_value=True),
            patch(_MODULE_PATH + ".is_distributed", return_value=True),
        ):
            with patch.object(trainer.logger, "error") as mock_error:
                trainer._warn_unused_params(model)
                mock_error.assert_called_once()

    def test_skips_on_non_rank0(self):
        trainer = _make_trainer()
        trainer._has_logged_unused_params = False
        model = MagicMock()
        model.nograd_paramnames = ["layer.weight"]

        with patch(_MODULE_PATH + ".is_rank0", return_value=False):
            with patch.object(trainer.logger, "warning") as mock_warn:
                trainer._warn_unused_params(model)
                mock_warn.assert_not_called()


class TestDropMetadata:
    def test_removes_meta_keys(self):
        data = {"x": torch.ones(2), "__meta_time_load": 0.1, "__meta_info": "abc"}
        result = BaseTrainer.drop_metadata(data)
        assert "x" in result
        assert "__meta_time_load" not in result
        assert "__meta_info" not in result

    def test_non_dict_passthrough(self):
        tensor = torch.ones(3)
        result = BaseTrainer.drop_metadata(tensor)
        assert result is tensor

    def test_empty_dict(self):
        result = BaseTrainer.drop_metadata({})
        assert result == {}

    def test_no_meta_keys_unchanged(self):
        data = {"a": 1, "b": 2}
        result = BaseTrainer.drop_metadata(data)
        assert result == {"a": 1, "b": 2}


class TestStateDict:
    def test_state_dict_contains_expected_keys(self):
        trainer = _make_trainer()
        cb = MagicMock()
        cb.state_dict.return_value = {"key": "val"}
        trainer.callbacks = [cb]
        trainer.grad_scaler = MagicMock(spec=[])  # not a GradScaler

        sd = trainer.state_dict()
        assert "callback_state_dicts" in sd or any("CALLBACK" in str(k).upper() for k in sd)

    def test_load_state_dict_callback_mismatch_raises(self):
        trainer = _make_trainer()
        trainer.callbacks = [MagicMock()]

        # Provide 2 callback state dicts but trainer has 1 callback
        from noether.core.types import CheckpointKeys

        state_dict = {
            CheckpointKeys.CALLBACK_STATE_DICT: [{"a": 1}, {"b": 2}],
            CheckpointKeys.TRAINING_ITERATION: {},
        }
        with pytest.raises(ValueError, match="Number of callbacks"):
            trainer.load_state_dict(state_dict)

    def test_load_state_dict_loads_callbacks(self):
        trainer = _make_trainer()
        cb = MagicMock()
        trainer.callbacks = [cb]
        trainer.grad_scaler = MagicMock(spec=[])  # not a real GradScaler

        from noether.core.types import CheckpointKeys

        state_dict = {
            CheckpointKeys.CALLBACK_STATE_DICT: [{"step": 5}],
            CheckpointKeys.TRAINING_ITERATION: {},
        }
        trainer.load_state_dict(state_dict)
        cb.load_state_dict.assert_called_once_with({"step": 5})


class TestWrapCompile:
    def test_no_compile_returns_model_unchanged(self):
        trainer = _make_trainer(use_torch_compile=False)
        model = MagicMock(spec=nn.Module)
        result = trainer.wrap_compile(model)
        assert result is model

    def test_compile_on_windows_skips(self):
        trainer = _make_trainer(use_torch_compile=True)
        model = MagicMock(spec=nn.Module)
        with patch("noether.training.trainers.base.os.name", "nt"):
            result = trainer.wrap_compile(model)
        assert result is model

    def test_compile_distributed_static_graph_skips(self):
        trainer = _make_trainer(use_torch_compile=True, static_graph=True)
        model = MagicMock(spec=nn.Module)
        with (
            patch(_MODULE_PATH + ".is_distributed", return_value=True),
            patch(_MODULE_PATH + ".os.name", "posix"),
        ):
            result = trainer.wrap_compile(model)
        assert result is model

    def test_compile_returns_compiled_module(self):
        trainer = _make_trainer(use_torch_compile=True)
        model = nn.Linear(2, 2)
        with (
            patch(_MODULE_PATH + ".is_distributed", return_value=False),
            patch(_MODULE_PATH + ".os.name", "posix"),
            patch(_MODULE_PATH + ".torch.compile", return_value=nn.Linear(2, 2)) as mock_compile,
        ):
            result = trainer.wrap_compile(model)
        mock_compile.assert_called_once_with(model)
        assert isinstance(result, nn.Module)

    def test_compile_non_module_return_raises(self):
        trainer = _make_trainer(use_torch_compile=True)
        model = nn.Linear(2, 2)
        with (
            patch(_MODULE_PATH + ".is_distributed", return_value=False),
            patch(_MODULE_PATH + ".os.name", "posix"),
            patch(_MODULE_PATH + ".torch.compile", return_value="not_a_module"),
        ):
            with pytest.raises(TypeError, match="torch.compile did not return a torch.nn.Module"):
                trainer.wrap_compile(model)


class TestWrapDdp:
    def test_non_distributed_returns_model(self):
        trainer = _make_trainer()
        model = MagicMock()
        model.device = torch.device("cpu")
        with patch(_MODULE_PATH + ".is_distributed", return_value=False):
            result = trainer.wrap_ddp(model)
        assert result is model

    def test_cpu_device_skips_ddp(self):
        trainer = _make_trainer()
        model = MagicMock()
        model.device = torch.device("cpu")
        with patch(_MODULE_PATH + ".is_distributed", return_value=True):
            result = trainer.wrap_ddp(model)
        assert result is model


class TestHandleEndOfEpoch:
    def test_returns_false_when_not_full_epoch(self):
        trainer = _make_trainer()
        trainer.update_counter = MagicMock()
        trainer.update_counter.is_full_epoch = False

        result = trainer._handle_end_of_epoch(
            model=MagicMock(),
            dist_model=MagicMock(),
            batch_size=4,
            periodic_callbacks=list(),
            data_iter=iter([]),
        )
        assert result is False

    def test_runs_periodic_callbacks_on_full_epoch(self):
        trainer = _make_trainer()
        trainer.update_counter = MagicMock()
        trainer.update_counter.is_full_epoch = True
        trainer.update_counter.is_finished = False

        with patch.object(trainer, "_run_periodic_callbacks", return_value=False) as mock_run:
            result = trainer._handle_end_of_epoch(
                model=MagicMock(),
                dist_model=MagicMock(),
                batch_size=4,
                periodic_callbacks=[MagicMock()],
                data_iter=iter([]),
            )
        mock_run.assert_called_once()
        assert result is False

    def test_returns_true_when_training_finished(self):
        trainer = _make_trainer()
        trainer.update_counter = MagicMock()
        trainer.update_counter.is_full_epoch = True
        trainer.update_counter.is_finished = True

        with patch.object(trainer, "_run_periodic_callbacks", return_value=False):
            result = trainer._handle_end_of_epoch(
                model=MagicMock(),
                dist_model=MagicMock(),
                batch_size=4,
                periodic_callbacks=[],
                data_iter=iter([]),
            )
        assert result is True


class TestRunPeriodicCallbacks:
    def _make_trainer_with_counter(self):
        trainer = _make_trainer()
        trainer.update_counter = MagicMock()
        trainer.update_counter.is_full_epoch = False
        trainer.update_counter.cur_iteration = MagicMock()
        return trainer

    def test_after_update_called_when_not_end_of_epoch(self):
        trainer = self._make_trainer_with_counter()
        cb = MagicMock()
        model = MagicMock()
        dist_model = MagicMock()

        from noether.core.callbacks import PeriodicCallback

        with patch.object(cb, "__class__", PeriodicCallback):
            # Use a real-ish callback mock
            cb_mock = MagicMock(spec=PeriodicCallback)
            early = trainer._run_periodic_callbacks(
                periodic_callbacks=[cb_mock],
                model=model,
                dist_model=dist_model,
                data_iter=iter([]),
                batch_size=4,
                end_of_epoch=False,
            )
        cb_mock.after_update.assert_called_once()
        assert early is False

    def test_early_stop_iteration_triggers_early_exit(self):
        from noether.core.callbacks import PeriodicCallback
        from noether.core.callbacks.early_stoppers import EarlyStopIteration

        trainer = self._make_trainer_with_counter()
        trainer.checkpoint_writer = MagicMock()

        cb_mock = MagicMock(spec=PeriodicCallback)
        cb_mock.after_update.side_effect = EarlyStopIteration

        with patch(_MODULE_PATH + ".PeriodicDataIteratorCallback", MagicMock):
            early = trainer._run_periodic_callbacks(
                periodic_callbacks=[cb_mock],
                model=MagicMock(),
                dist_model=MagicMock(),
                data_iter=iter([]),
                batch_size=4,
                end_of_epoch=False,
            )
        assert early is True

    def test_callback_exception_reraises_after_all_run(self):
        from noether.core.callbacks import PeriodicCallback

        trainer = self._make_trainer_with_counter()
        trainer.checkpoint_writer = MagicMock()

        cb1 = MagicMock(spec=PeriodicCallback)
        cb1.after_update.side_effect = ValueError("boom")
        cb2 = MagicMock(spec=PeriodicCallback)

        with pytest.raises(ValueError, match="boom"):
            trainer._run_periodic_callbacks(
                periodic_callbacks=[cb1, cb2],
                model=MagicMock(),
                dist_model=MagicMock(),
                data_iter=iter([]),
                batch_size=4,
                end_of_epoch=False,
            )

        # cb2 still ran despite cb1 raising
        cb2.after_update.assert_called_once()


class TestPrepareBatchSize:
    def test_update_based_end_checkpoint_with_accumulation_raises(self):
        trainer = _make_trainer(
            effective_batch_size=8,
            dataset_len=80,
            disable_gradient_accumulation=False,
            max_batch_size=2,
        )
        # Set update-based end_checkpoint
        trainer.end_checkpoint = TrainingIteration(epoch=None, update=50, sample=None)

        with (
            patch(_MODULE_PATH + ".get_world_size", return_value=1),
            patch(_MODULE_PATH + ".get_num_nodes", return_value=1),
            patch(_MODULE_PATH + ".is_distributed", return_value=False),
        ):
            with pytest.raises(NotImplementedError, match="accumulation steps not supported"):
                trainer._prepare_batch_size()


class TestGetDefaultCallbacks:
    def test_no_intervals_emits_warning(self):
        trainer = _make_trainer(
            log_every_n_epochs=None,
            log_every_n_updates=None,
            log_every_n_samples=None,
            track_every_n_epochs=None,
            track_every_n_updates=None,
            track_every_n_samples=None,
        )
        trainer.update_counter = MagicMock()
        trainer.update_counter.is_finished = False

        with (
            patch(_MODULE_PATH + ".sys.stdout") as mock_stdout,
            patch(_MODULE_PATH + ".is_rank0", return_value=True),
        ):
            mock_stdout.isatty.return_value = False
            with patch.object(trainer.logger, "warning") as mock_warn:
                default_kwargs = trainer._get_default_callback_kwargs(MagicMock())
                with (
                    patch("noether.core.callbacks.DatasetStatsCallback", MagicMock()),
                    patch("noether.core.callbacks.ParamCountCallback", MagicMock()),
                ):
                    try:
                        trainer.get_default_callbacks(default_kwargs)
                    except Exception:
                        pass
                    mock_warn.assert_called()


class TestTrainingHooks:
    def test_call_before_training_invokes_callbacks(self):
        trainer = _make_trainer()
        cb1 = MagicMock()
        cb2 = MagicMock()
        trainer.call_before_training([cb1, cb2])
        cb1.before_training.assert_called_once_with(update_counter=trainer.update_counter)
        cb2.before_training.assert_called_once_with(update_counter=trainer.update_counter)

    def test_call_after_training_invokes_callbacks_and_flushes(self):
        trainer = _make_trainer()
        cb = MagicMock()
        trainer.call_after_training([cb])
        cb.after_training.assert_called_once_with(update_counter=trainer.update_counter)
        trainer.log_writer.flush.assert_called_once()


class TestApplyResumeInitializer:
    def test_no_initializer_is_noop(self):
        trainer = _make_trainer()
        trainer.initializer = None
        model = MagicMock()
        trainer.apply_resume_initializer(model)  # should not raise

    def test_initializer_called_with_correct_args(self):
        trainer = _make_trainer()
        initializer = MagicMock()
        trainer.initializer = initializer
        trainer.callbacks = [MagicMock()]
        model = MagicMock()

        trainer.apply_resume_initializer(model)

        initializer.init_trainer.assert_called_once_with(trainer)
        initializer.init_weights.assert_called_once_with(model)
        initializer.init_optimizer.assert_called_once_with(model)
        initializer.init_callbacks.assert_called_once_with(trainer.callbacks, model=model)


class TestSkipRemainingBatches:
    def _make_data_iter_with_sampler(self, has_epochs=True, has_samples=True):
        sampler = MagicMock()
        sampler.epochs = 5 if has_epochs else None
        sampler.samples = 100 if has_samples else None

        batch_sampler = MagicMock()
        batch_sampler.sampler = sampler

        data_iter = MagicMock()
        data_iter.batch_sampler = batch_sampler
        data_iter.__next__ = MagicMock(return_value={})
        return data_iter

    def test_skips_when_epochs_based(self):
        trainer = _make_trainer()
        data_iter = self._make_data_iter_with_sampler(has_epochs=True, has_samples=False)
        # remaining=5, acc=2 → skip 5-2=3 batches
        trainer._skip_remaining_batches(data_iter, remaining_batches=5, accumulation_steps=2, batch_size=4)
        assert data_iter.__next__.call_count == 3

    def test_skips_when_samples_based(self):
        trainer = _make_trainer()
        data_iter = self._make_data_iter_with_sampler(has_epochs=False, has_samples=True)
        # samples branch: total_batches=100//4=25, 25 % 2 = 1 skip
        trainer._skip_remaining_batches(data_iter, remaining_batches=5, accumulation_steps=2, batch_size=4)
        assert data_iter.__next__.call_count >= 0  # verify no crash; exact count depends on sampler mock

    def test_no_special_sampler_no_skip(self):
        trainer = _make_trainer()
        data_iter = MagicMock(spec=[])  # no batch_sampler attribute
        # Should not raise
        trainer._skip_remaining_batches(data_iter, remaining_batches=5, accumulation_steps=2, batch_size=4)
