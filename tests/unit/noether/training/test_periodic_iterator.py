#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from unittest.mock import Mock, patch

import pytest
import torch

from noether.core.callbacks.periodic import PeriodicDataIteratorCallback
from noether.core.schemas.callbacks import PeriodicDataIteratorCallbackConfig
from noether.core.utils.training import UpdateCounter


@pytest.fixture
def mock_trainer():
    """Mock SgdTrainer instance."""
    trainer = Mock()
    trainer.update_counter = Mock(spec=UpdateCounter)
    trainer.device = "cpu"
    trainer.dataset_mode = {"train"}
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
    data_container = Mock()
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=100)
    mock_dataset.collator = None
    data_container.get_dataset = Mock(return_value=mock_dataset)
    return data_container


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


class TestPeriodicDataIteratorCallback:
    """Tests for PeriodicDataIteratorCallback."""

    def test_instantiation(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test that PeriodicDataIteratorCallback can be instantiated."""
        config = PeriodicDataIteratorCallbackConfig.model_validate(dict(every_n_updates=10, dataset_key="test"))

        class TestCallback(PeriodicDataIteratorCallback):
            def process_data(self, batch, **_):
                pass

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

        assert callback is not None
        assert callback.sampler_config is not None

    @patch("noether.core.callbacks.periodic.is_distributed")
    @patch("noether.core.callbacks.periodic.is_rank0")
    def test_iterate_over_dataset(
        self,
        mock_is_rank0,
        mock_is_distributed,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test iterating over a dataset."""
        mock_is_distributed.return_value = False
        mock_is_rank0.return_value = True

        config = PeriodicDataIteratorCallbackConfig.model_validate(
            dict(every_n_updates=10, batch_size=4, dataset_key="test")
        )

        class TestCallback(PeriodicDataIteratorCallback):
            def process_data(self, batch, *, trainer_model):
                return {"result": torch.tensor([1.0])}

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

        # Mock data iterator
        mock_batch = {"x": torch.randn(4, 10)}
        mock_data_iter = iter([mock_batch] * 25)  # 25 batches for 100 samples

        result = callback._iterate_over_dataset(
            batch_size=4,
            data_iter=mock_data_iter,
            trainer_model=mock_model,
        )

        assert result is not None
        assert "result" in result

    def test_collate_tensors(self):
        """Test tensor collation."""
        # Test collating 0-dim tensors (scalars)
        tensors_0d = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        result_0d = PeriodicDataIteratorCallback._collate_tensors(tensors_0d)
        assert result_0d.shape == (3,)
        assert torch.allclose(result_0d, torch.tensor([1.0, 2.0, 3.0]))

        # Test collating 1-dim tensors
        tensors_1d = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        result_1d = PeriodicDataIteratorCallback._collate_tensors(tensors_1d)
        assert result_1d.shape == (4,)
        assert torch.allclose(result_1d, torch.tensor([1.0, 2.0, 3.0, 4.0]))

    @patch("noether.core.callbacks.periodic.is_distributed")
    @patch("noether.core.callbacks.periodic.is_rank0")
    def testperiodic_callback_single_output(
        self,
        mock_is_rank0,
        mock_is_distributed,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test periodic_callback with single output from process_data."""
        mock_is_distributed.return_value = False
        mock_is_rank0.return_value = True

        config = PeriodicDataIteratorCallbackConfig.model_validate(
            dict(every_n_updates=10, batch_size=4, dataset_key="test")
        )

        processed_results = []

        class TestCallback(PeriodicDataIteratorCallback):
            def process_data(self, batch, *, trainer_model):
                return {"loss": torch.tensor([1.0, 2.0, 3.0, 4.0])}

            def process_results(self, results, *, interval_type, update_counter, **_):
                processed_results.append(results)

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

        # Mock data iterator
        mock_batch = {"x": torch.randn(4, 10)}
        mock_data_iter = iter([mock_batch] * 25)

        # Mock update counter
        mock_update_counter = Mock(spec=UpdateCounter)

        callback.periodic_callback(
            interval_type="update",
            update_counter=mock_update_counter,
            data_iter=mock_data_iter,
            trainer_model=mock_model,
            batch_size=4,
        )

        assert len(processed_results) == 1
        assert "loss" in processed_results[0]

    @patch("noether.core.callbacks.periodic.is_distributed")
    @patch("noether.core.callbacks.periodic.is_rank0")
    def testperiodic_callback_multiple_outputs(
        self,
        mock_is_rank0,
        mock_is_distributed,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test periodic_callback with multiple outputs from process_data."""
        mock_is_distributed.return_value = False
        mock_is_rank0.return_value = True

        config = PeriodicDataIteratorCallbackConfig.model_validate(
            dict(every_n_updates=10, batch_size=4, dataset_key="test")
        )

        processed_results = []

        class TestCallback(PeriodicDataIteratorCallback):
            def process_data(self, batch, *, trainer_model):
                predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
                labels = torch.tensor([0, 1, 0, 1])
                return predictions, labels

            def process_results(self, results, *, interval_type, update_counter, **_):
                processed_results.append(results)

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

        # Mock data iterator
        mock_batch = {"x": torch.randn(4, 10)}
        mock_data_iter = iter([mock_batch] * 25)

        # Mock update counter
        mock_update_counter = Mock(spec=UpdateCounter)

        callback.periodic_callback(
            interval_type="update",
            update_counter=mock_update_counter,
            data_iter=mock_data_iter,
            trainer_model=mock_model,
            batch_size=4,
        )

        assert len(processed_results) == 1
        predictions, labels = processed_results[0]
        assert predictions is not None
        assert labels is not None

    @patch("noether.core.callbacks.periodic.is_distributed")
    @patch("noether.core.callbacks.periodic.is_rank0")
    def testperiodic_callback_passes_correct_arguments(
        self,
        mock_is_rank0,
        mock_is_distributed,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test that periodic_callback passes correct arguments to process_results."""
        mock_is_distributed.return_value = False
        mock_is_rank0.return_value = True

        config = PeriodicDataIteratorCallbackConfig.model_validate(
            dict(every_n_updates=10, batch_size=4, dataset_key="test")
        )

        process_results_calls = []

        class TestCallback(PeriodicDataIteratorCallback):
            def process_data(self, batch, *, trainer_model):
                return {"data": torch.tensor([1.0])}

            def process_results(self, results, *, interval_type, update_counter, **_):
                process_results_calls.append(
                    {
                        "results": results,
                        "interval_type": interval_type,
                        "update_counter": update_counter,
                    }
                )

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

        # Mock data iterator
        mock_batch = {"x": torch.randn(4, 10)}
        mock_data_iter = iter([mock_batch] * 25)

        # Mock update counter
        mock_update_counter = Mock(spec=UpdateCounter)

        callback.periodic_callback(
            interval_type="epoch",
            update_counter=mock_update_counter,
            data_iter=mock_data_iter,
            trainer_model=mock_model,
            batch_size=4,
        )

        assert len(process_results_calls) == 1
        call = process_results_calls[0]
        assert call["interval_type"] == "epoch"
        assert call["update_counter"] is mock_update_counter
        assert "data" in call["results"]
