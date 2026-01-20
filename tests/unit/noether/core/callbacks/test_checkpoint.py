#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from unittest.mock import Mock

import pytest

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


class TestCheckpointCallbacks:
    """Tests for checkpoint callbacks."""

    def test_checkpoint_callback_saves_checkpoint(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test CheckpointCallback saves checkpoints periodically."""
        from noether.core.callbacks.checkpoint import CheckpointCallback
        from noether.core.schemas.callbacks import CheckpointCallbackConfig

        config = CheckpointCallbackConfig.model_validate(dict(every_n_updates=10))
        mock_trainer.update_counter = Mock()
        mock_trainer.update_counter.update = 10

        callback = CheckpointCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        callback._periodic_callback(
            interval_type="update", trainer=mock_trainer, update_counter=mock_trainer.update_counter
        )
        mock_checkpoint_writer.save.assert_called()
