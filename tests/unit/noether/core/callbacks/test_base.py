#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from noether.core.callbacks.base import CallbackBase
from noether.core.providers.path import PathProvider
from noether.core.utils.training.counter import UpdateCounter


@pytest.fixture
def mock_trainer():
    """Mock SgdTrainer instance."""
    trainer = Mock()
    trainer.update_counter = Mock(spec=UpdateCounter)
    return trainer


@pytest.fixture
def mock_model():
    """Mock ModelBase instance."""
    model = Mock()
    model.eval = Mock()
    model.train = Mock()
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


@pytest.fixture
def callback_base(
    mock_trainer,
    mock_model,
    mock_data_container,
    mock_tracker,
    mock_log_writer,
    mock_checkpoint_writer,
    mock_metric_property_provider,
):
    """Create a CallbackBase instance with all mocked dependencies."""
    return CallbackBase(
        trainer=mock_trainer,
        model=mock_model,
        data_container=mock_data_container,
        tracker=mock_tracker,
        log_writer=mock_log_writer,
        checkpoint_writer=mock_checkpoint_writer,
        metric_property_provider=mock_metric_property_provider,
        name="test_callback",
    )


class TestCallbackBase:
    """Smoke tests for CallbackBase."""

    def test_instantiation(self, callback_base):
        """Test that CallbackBase can be instantiated."""
        assert callback_base is not None
        assert callback_base.name == "test_callback"

    def test_instantiation_without_name(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test that CallbackBase can be instantiated without a name."""
        callback = CallbackBase(
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )
        assert callback.name is None

    def test_logger_property(self, callback_base):
        """Test that logger property returns a logger instance."""
        logger = callback_base.logger
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")

    def test_state_dict_default(self, callback_base):
        """Test that default state_dict returns None."""
        state = callback_base.state_dict()
        assert state is None

    def test_load_state_dict_default(self, callback_base):
        """Test that default load_state_dict doesn't raise errors."""
        callback_base.load_state_dict(None)
        callback_base.load_state_dict({"key": torch.tensor([1.0])})

    def test_resume_from_checkpoint_default(self, callback_base, mock_model):
        """Test that default resume_from_checkpoint doesn't raise errors."""
        callback_base.resume_from_checkpoint(PathProvider(Path("root"), "run_id", None), mock_model)

    def test_before_training_hook(self, callback_base):
        """Test that before_training can be called without errors."""
        update_counter = Mock(spec=UpdateCounter)
        callback_base.before_training(update_counter=update_counter)

    def test_after_training_hook(self, callback_base):
        """Test that after_training can be called without errors."""
        update_counter = Mock(spec=UpdateCounter)
        callback_base.after_training(update_counter=update_counter)

    def test_custom_implementation_methods(self):
        """Test that custom implementations of template methods work correctly."""

        class CustomCallback(CallbackBase):
            def before_training(self, **_):
                self.before_called = True

            def after_training(self, **_):
                self.after_called = True

        custom_callback = CustomCallback(
            trainer=Mock(),
            model=Mock(),
            data_container=Mock(),
            tracker=Mock(),
            log_writer=Mock(),
            checkpoint_writer=Mock(),
            metric_property_provider=Mock(),
        )

        update_counter = Mock(spec=UpdateCounter)

        custom_callback.before_training(update_counter=update_counter)
        assert hasattr(custom_callback, "before_called")
        assert custom_callback.before_called is True

        custom_callback.after_training(update_counter=update_counter)
        assert hasattr(custom_callback, "after_called")
        assert custom_callback.after_called is True
