#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from noether.core.callbacks.early_stoppers.metric import MetricEarlyStopper, MetricEarlyStopperConfig


class TestMetricEarlyStopper:
    """Test suite for MetricEarlyStopper."""

    @pytest.fixture
    def mock_trainer(self):
        """Mock SgdTrainer instance."""
        trainer = Mock()
        trainer.update_counter = Mock()
        return trainer

    @pytest.fixture
    def mock_model(self):
        """Mock ModelBase instance."""
        model = Mock()
        model.eval = Mock()
        model.train = Mock()
        return model

    @pytest.fixture
    def mock_data_container(self):
        """Mock DataContainer instance."""
        return Mock()

    @pytest.fixture
    def mock_tracker(self):
        """Mock BaseTracker instance."""
        tracker = Mock()
        tracker.log_scalar = Mock()
        return tracker

    @pytest.fixture
    def mock_log_writer(self):
        """Mock LogWriter instance."""
        writer = Mock()
        writer.add_scalar = Mock()
        writer.log_cache = {}
        return writer

    @pytest.fixture
    def mock_checkpoint_writer(self):
        """Mock CheckpointWriter instance."""
        writer = Mock()
        writer.save = Mock()
        return writer

    @pytest.fixture
    def mock_metric_property_provider(self):
        """Mock MetricPropertyProvider instance."""
        provider = Mock()
        provider.higher_is_better = Mock(return_value=True)
        return provider

    def test_init_valid_tolerance(
        self,
        mock_metric_property_provider,
        mock_log_writer,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_checkpoint_writer,
    ):
        """Test initialization with valid tolerance."""
        stopper = MetricEarlyStopper(
            callback_config=MetricEarlyStopperConfig(metric_key="accuracy", tolerance=3, every_n_updates=1),
            model=mock_model,
            trainer=mock_trainer,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )
        assert stopper.metric_key == "accuracy"
        assert stopper.tolerance == 3
        assert stopper.tolerance_counter == 0
        assert stopper.higher_is_better is True
        assert stopper.best_metric == -float("inf")

    def test_init_invalid_tolerance(
        self,
        mock_metric_property_provider,
        mock_log_writer,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_checkpoint_writer,
    ):
        """Test initialization with invalid tolerance."""
        with pytest.raises(ValidationError):
            MetricEarlyStopper(
                callback_config=MetricEarlyStopperConfig(metric_key="accuracy", tolerance=0, every_n_updates=1),
                model=mock_model,
                trainer=mock_trainer,
                data_container=mock_data_container,
                tracker=mock_tracker,
                log_writer=mock_log_writer,
                checkpoint_writer=mock_checkpoint_writer,
                metric_property_provider=mock_metric_property_provider,
            )

    def test_metric_improved_higher_is_better(
        self,
        mock_metric_property_provider,
        mock_log_writer,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_checkpoint_writer,
    ):
        """Test metric improvement when higher is better."""
        stopper = MetricEarlyStopper(
            callback_config=MetricEarlyStopperConfig(metric_key="accuracy", tolerance=3, every_n_updates=1),
            model=mock_model,
            trainer=mock_trainer,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )
        stopper.best_metric = 0.8
        assert stopper._metric_improved(0.85) is True
        assert stopper._metric_improved(0.75) is False
        assert stopper._metric_improved(0.8) is False

    def test_metric_improved_lower_is_better(
        self,
        mock_metric_property_provider,
        mock_log_writer,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_checkpoint_writer,
    ):
        """Test metric improvement when lower is better."""
        mock_metric_property_provider.higher_is_better = Mock(return_value=False)
        stopper = MetricEarlyStopper(
            callback_config=MetricEarlyStopperConfig(metric_key="loss", tolerance=3, every_n_updates=1),
            model=mock_model,
            trainer=mock_trainer,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )
        stopper.best_metric = 0.5
        assert stopper._metric_improved(0.4) is True
        assert stopper._metric_improved(0.6) is False
        assert stopper._metric_improved(0.5) is False

    def test_should_stop_metric_not_found(
        self,
        mock_metric_property_provider,
        mock_log_writer,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_checkpoint_writer,
    ):
        """Test error when metric key is not found."""
        mock_log_writer.log_cache = {"loss": 0.5}
        stopper = MetricEarlyStopper(
            callback_config=MetricEarlyStopperConfig(metric_key="accuracy", tolerance=3, every_n_updates=1),
            model=mock_model,
            trainer=mock_trainer,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )
        with pytest.raises(ValueError, match="couldn't find metric_key accuracy"):
            stopper._should_stop()

    def test_should_stop_metric_improves(
        self,
        mock_metric_property_provider,
        mock_log_writer,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_checkpoint_writer,
    ):
        """Test no stopping when metric improves."""
        mock_log_writer.log_cache = {"accuracy": 0.85}
        stopper = MetricEarlyStopper(
            callback_config=MetricEarlyStopperConfig(metric_key="accuracy", tolerance=3, every_n_updates=1),
            model=mock_model,
            trainer=mock_trainer,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )
        stopper.best_metric = 0.8

        assert stopper._should_stop() is False
        assert stopper.best_metric == 0.85
        assert stopper.tolerance_counter == 0

    def test_should_stop_metric_stagnates_within_tolerance(
        self,
        mock_metric_property_provider,
        mock_log_writer,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_checkpoint_writer,
    ):
        """Test no stopping when metric stagnates but within tolerance."""
        mock_log_writer.log_cache = {"accuracy": 0.75}
        stopper = MetricEarlyStopper(
            callback_config=MetricEarlyStopperConfig(metric_key="accuracy", tolerance=3, every_n_updates=1),
            model=mock_model,
            trainer=mock_trainer,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )
        stopper.best_metric = 0.8

        assert stopper._should_stop() is False
        assert stopper.best_metric == 0.8
        assert stopper.tolerance_counter == 1

    def test_should_stop_metric_stagnates_exceeds_tolerance(
        self,
        mock_metric_property_provider,
        mock_log_writer,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_checkpoint_writer,
    ):
        """Test stopping when metric stagnates and exceeds tolerance."""
        mock_log_writer.log_cache = {"accuracy": 0.75}
        stopper = MetricEarlyStopper(
            callback_config=MetricEarlyStopperConfig(metric_key="accuracy", tolerance=3, every_n_updates=1),
            model=mock_model,
            trainer=mock_trainer,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )
        stopper.best_metric = 0.8
        stopper.tolerance_counter = 2

        assert stopper._should_stop() is True
        assert stopper.tolerance_counter == 3

    def test_should_stop_tolerance_counter_resets_on_improvement(
        self,
        mock_metric_property_provider,
        mock_log_writer,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_checkpoint_writer,
    ):
        """Test tolerance counter resets when metric improves."""
        stopper = MetricEarlyStopper(
            callback_config=MetricEarlyStopperConfig(metric_key="accuracy", tolerance=3, every_n_updates=1),
            model=mock_model,
            trainer=mock_trainer,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )
        stopper.best_metric = 0.8
        stopper.tolerance_counter = 2

        # Metric improves
        mock_log_writer.log_cache = {"accuracy": 0.85}
        assert stopper._should_stop() is False
        assert stopper.tolerance_counter == 0

        # Metric stagnates again
        mock_log_writer.log_cache = {"accuracy": 0.82}
        assert stopper._should_stop() is False
        assert stopper.tolerance_counter == 1

    def test_should_stop_sequence(
        self,
        mock_metric_property_provider,
        mock_log_writer,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_checkpoint_writer,
    ):
        """Test a sequence of metric updates leading to early stopping."""
        mock_metric_property_provider.higher_is_better = Mock(return_value=False)
        stopper = MetricEarlyStopper(
            callback_config=MetricEarlyStopperConfig(metric_key="loss", tolerance=2, every_n_updates=1),
            model=mock_model,
            trainer=mock_trainer,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        # First improvement
        mock_log_writer.log_cache = {"loss": 0.5}
        assert stopper._should_stop() is False
        assert stopper.tolerance_counter == 0

        # Second improvement
        mock_log_writer.log_cache = {"loss": 0.4}
        assert stopper._should_stop() is False
        assert stopper.tolerance_counter == 0

        # First stagnation
        mock_log_writer.log_cache = {"loss": 0.45}
        assert stopper._should_stop() is False
        assert stopper.tolerance_counter == 1

        # Second stagnation - should stop
        mock_log_writer.log_cache = {"loss": 0.5}
        assert stopper._should_stop() is True
        assert stopper.tolerance_counter == 2

    def test_should_stop_log_cache_none(
        self,
        mock_metric_property_provider,
        mock_log_writer,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_checkpoint_writer,
    ):
        """Test error when log_cache is None."""
        mock_log_writer.log_cache = None
        stopper = MetricEarlyStopper(
            callback_config=MetricEarlyStopperConfig(metric_key="accuracy", tolerance=3, every_n_updates=1),
            model=mock_model,
            trainer=mock_trainer,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )
        with pytest.raises(ValueError, match="couldn't find metric_key accuracy"):
            stopper._should_stop()
