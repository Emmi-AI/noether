#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from unittest.mock import Mock

import pytest
import torch

from noether.core.callbacks.periodic import PeriodicCallback
from noether.core.schemas.callbacks import CallBackBaseConfig
from noether.core.utils.training import TrainingIteration, UpdateCounter


@pytest.fixture
def mock_trainer():
    """Mock SgdTrainer instance."""
    trainer = Mock()
    trainer.update_counter = Mock(spec=UpdateCounter)
    trainer.device = "cpu"
    return trainer


@pytest.fixture
def mock_model():
    """Mock ModelBase instance."""
    model = Mock()
    model.eval = Mock()
    model.train = Mock()
    model.device = "cpu"
    return model


@pytest.fixture
def mock_data_container():
    """Mock DataContainer instance."""
    return Mock()


@pytest.fixture
def mock_tracker():
    """Mock BaseTracker instance."""
    tracker = Mock()
    tracker.log_scalar = Mock()
    return tracker


@pytest.fixture
def mock_log_writer():
    """Mock LogWriter instance."""
    writer = Mock()
    writer.add_scalar = Mock()
    return writer


@pytest.fixture
def mock_checkpoint_writer():
    """Mock CheckpointWriter instance."""
    writer = Mock()
    writer.save = Mock()
    return writer


@pytest.fixture
def mock_metric_property_provider():
    """Mock MetricPropertyProvider instance."""
    return Mock()


class TestPeriodicCallback:
    """Tests for PeriodicCallback."""

    def test_instantiation_with_every_n_epochs(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test PeriodicCallback instantiation with epoch-based interval."""
        config = CallBackBaseConfig.model_validate(dict(every_n_epochs=5))

        callback = PeriodicCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        assert callback.every_n_epochs == 5
        assert callback.every_n_updates is None
        assert callback.every_n_samples is None

    def test_instantiation_with_every_n_updates(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test PeriodicCallback instantiation with update-based interval."""
        config = CallBackBaseConfig.model_validate(dict(every_n_updates=100))

        callback = PeriodicCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        assert callback.every_n_epochs is None
        assert callback.every_n_updates == 100
        assert callback.every_n_samples is None

    def test_instantiation_with_every_n_samples(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test PeriodicCallback instantiation with sample-based interval."""
        config = CallBackBaseConfig.model_validate(dict(every_n_samples=1000, batch_size=4))

        callback = PeriodicCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        assert callback.every_n_epochs is None
        assert callback.every_n_updates is None
        assert callback.every_n_samples == 1000

    def test_should_log_after_epoch(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test should_log_after_epoch returns True at correct intervals."""
        config = CallBackBaseConfig.model_validate(dict(every_n_epochs=2))

        callback = PeriodicCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        checkpoint1 = TrainingIteration(epoch=1, update=10, sample=1000)
        checkpoint2 = TrainingIteration(epoch=2, update=20, sample=2000)
        checkpoint3 = TrainingIteration(epoch=3, update=30, sample=3000)
        checkpoint4 = TrainingIteration(epoch=4, update=40, sample=4000)

        assert callback._should_invoke_after_epoch(checkpoint1) is False
        assert callback._should_invoke_after_epoch(checkpoint2) is True
        assert callback._should_invoke_after_epoch(checkpoint3) is False
        assert callback._should_invoke_after_epoch(checkpoint4) is True

    def test_should_log_after_update(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test should_log_after_update returns True at correct intervals."""
        config = CallBackBaseConfig.model_validate(dict(every_n_updates=10))

        callback = PeriodicCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        checkpoint1 = TrainingIteration(epoch=1, update=5, sample=500)
        checkpoint2 = TrainingIteration(epoch=1, update=10, sample=1000)
        checkpoint3 = TrainingIteration(epoch=1, update=15, sample=1500)
        checkpoint4 = TrainingIteration(epoch=2, update=20, sample=2000)

        assert callback._should_invoke_after_update(checkpoint1) is False
        assert callback._should_invoke_after_update(checkpoint2) is True
        assert callback._should_invoke_after_update(checkpoint3) is False
        assert callback._should_invoke_after_update(checkpoint4) is True

    def test_should_log_after_sample(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test should_log_after_sample returns True at correct intervals."""
        config = CallBackBaseConfig.model_validate(dict(every_n_samples=1000, batch_size=4))

        callback = PeriodicCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        effective_batch_size = 100

        checkpoint1 = TrainingIteration(epoch=1, update=5, sample=500)
        checkpoint2 = TrainingIteration(epoch=1, update=10, sample=1000)
        checkpoint3 = TrainingIteration(epoch=1, update=15, sample=1500)
        checkpoint4 = TrainingIteration(epoch=2, update=20, sample=2000)

        assert callback._should_invoke_after_sample(checkpoint1, effective_batch_size) is False
        assert callback._should_invoke_after_sample(checkpoint2, effective_batch_size) is True
        assert callback._should_invoke_after_sample(checkpoint3, effective_batch_size) is False
        assert callback._should_invoke_after_sample(checkpoint4, effective_batch_size) is True

    def test_updates_per_log_interval_updates(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test updates_per_log_interval calculation for update-based intervals."""
        config = CallBackBaseConfig.model_validate(dict(every_n_updates=25))

        callback = PeriodicCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        mock_update_counter = Mock(spec=UpdateCounter)

        assert callback.updates_per_interval(mock_update_counter) == 25

    def test_updates_till_next_log(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test updates_till_next_log calculation."""
        config = CallBackBaseConfig.model_validate(dict(every_n_updates=10))

        callback = PeriodicCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        mock_update_counter = Mock(spec=UpdateCounter)
        mock_update_counter.cur_iteration = TrainingIteration(epoch=1, update=3, sample=300)

        assert callback.updates_till_next_invocation(mock_update_counter) == 7

    def test_get_interval_string_verbose(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test get_interval_string_verbose returns correct string representation."""
        config = CallBackBaseConfig.model_validate(dict(every_n_updates=100))

        callback = PeriodicCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        assert callback.get_interval_string_verbose() == "every_n_updates=100"

    def test_to_short_interval_string(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test to_short_interval_string returns correct short representation."""
        config = CallBackBaseConfig.model_validate(dict(every_n_epochs=5))

        callback = PeriodicCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        assert callback.to_short_interval_string() == "E5"

    def test_track_after_accumulation_step(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test track_after_accumulation_step can be called without errors."""
        config = CallBackBaseConfig.model_validate(dict(every_n_updates=10))

        callback = PeriodicCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        mock_update_counter = Mock(spec=UpdateCounter)
        batch = {"x": torch.randn(4, 10)}
        losses = {"total": torch.tensor(0.5)}
        update_outputs = {"grad_norm": torch.tensor(1.0)}

        callback.track_after_accumulation_step(
            update_counter=mock_update_counter,
            batch=batch,
            losses=losses,
            update_outputs=update_outputs,
            accumulation_steps=4,
            accumulation_step=1,
        )

    def testtrack_after_update_step(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test track_after_update_step can be called without errors."""
        config = CallBackBaseConfig.model_validate(dict(every_n_updates=10))

        callback = PeriodicCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        mock_update_counter = Mock(spec=UpdateCounter)
        times = {"forward": 0.1, "backward": 0.2}

        callback.track_after_update_step(update_counter=mock_update_counter, times=times)

    def test_customperiodic_callback_implementation(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test that custom periodic_callback implementation works."""
        config = CallBackBaseConfig.model_validate(dict(every_n_updates=5))

        class TestCallback(PeriodicCallback):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.callback_called = False

            def periodic_callback(self, **kwargs):
                self.callback_called = True

        callback = TestCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        mock_update_counter = Mock(spec=UpdateCounter)
        mock_update_counter.cur_iteration = TrainingIteration(epoch=1, update=5, sample=500)
        mock_update_counter.effective_batch_size = 100
        mock_data_iter = Mock()

        callback.after_update(
            trainer_model=mock_model,
            update_counter=mock_update_counter,
            data_iter=mock_data_iter,
            batch_size=32,
        )

        assert callback.callback_called is True
