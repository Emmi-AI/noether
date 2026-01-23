#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from unittest.mock import Mock

import pytest
import torch

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


class TestOnlineLossCallback:
    """Tests for OnlineLossCallback."""

    def test_online_loss_callback_tracks_loss(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test OnlineLossCallback tracks and logs loss during training."""
        from noether.core.callbacks.default import OnlineLossCallback
        from noether.core.schemas.callbacks import OnlineLossCallbackConfig

        config = OnlineLossCallbackConfig.model_validate(dict(every_n_updates=1, verbose=True))
        mock_trainer.update_counter = Mock()
        mock_trainer.update_counter.update = 1
        mock_trainer.config = Mock()
        mock_trainer.config.skip_nan_loss = False

        callback = OnlineLossCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        losses = {"total": torch.tensor(0.5)}
        callback.track_after_accumulation_step(losses=losses)

        assert "total" in callback.tracked_losses
        assert len(callback.tracked_losses["total"]) == 1

    def test_online_loss_callback_detects_nan(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test OnlineLossCallback detects NaN losses."""
        from noether.core.callbacks.default import OnlineLossCallback
        from noether.core.schemas.callbacks import OnlineLossCallbackConfig

        config = OnlineLossCallbackConfig.model_validate(dict(every_n_updates=1, verbose=True))
        mock_trainer.update_counter = Mock()
        mock_trainer.update_counter.update = 1
        mock_trainer.config = Mock()
        mock_trainer.config.skip_nan_loss = False

        callback = OnlineLossCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        losses = {"total": torch.tensor(float("nan"))}
        callback.track_after_accumulation_step(losses=losses)

        with pytest.raises(RuntimeError, match="encountered nan loss"):
            callback.periodic_callback(interval_type="update", trainer=mock_trainer)


class TestBestMetricCallback:
    """Tests for BestMetricCallback."""

    def test_best_metric_callback_instantiation_higher_is_better(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test BestMetricCallback instantiates correctly for higher-is-better metrics."""
        from noether.core.callbacks.online.best_metric import BestMetricCallback
        from noether.core.schemas.callbacks import BestMetricCallbackConfig

        mock_metric_property_provider.higher_is_better = Mock(return_value=True)

        config = BestMetricCallbackConfig.model_validate(
            dict(
                source_metric_key="val/accuracy",
                target_metric_keys=["test/accuracy"],
                every_n_updates=1,
            )
        )

        callback = BestMetricCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        assert callback.source_metric_key == "val/accuracy"
        assert callback.target_metric_keys == ["test/accuracy"]
        assert callback.higher_is_better is True
        assert callback.best_metric_value == -float("inf")

    def test_best_metric_callback_instantiation_lower_is_better(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test BestMetricCallback instantiates correctly for lower-is-better metrics."""
        from noether.core.callbacks.online.best_metric import BestMetricCallback
        from noether.core.schemas.callbacks import BestMetricCallbackConfig

        mock_metric_property_provider.higher_is_better = Mock(return_value=False)

        config = BestMetricCallbackConfig.model_validate(
            dict(
                source_metric_key="val/loss",
                target_metric_keys=["test/loss"],
                every_n_updates=1,
            )
        )

        callback = BestMetricCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        assert callback.higher_is_better is False
        assert callback.best_metric_value == float("inf")

    def test_best_metric_callback_tracks_new_best_higher(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test BestMetricCallback tracks new best when metric improves (higher is better)."""
        from noether.core.callbacks.online.best_metric import BestMetricCallback
        from noether.core.schemas.callbacks import BestMetricCallbackConfig

        mock_metric_property_provider.higher_is_better = Mock(return_value=True)
        mock_log_writer.log_cache = {"val/accuracy": 0.9, "test/accuracy": 0.85}

        config = BestMetricCallbackConfig.model_validate(
            dict(
                source_metric_key="val/accuracy",
                target_metric_keys=["test/accuracy"],
                every_n_updates=1,
            )
        )

        callback = BestMetricCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        callback.periodic_callback(trainer=mock_trainer)

        assert callback.best_metric_value == 0.9
        mock_log_writer.add_scalar.assert_called()

    def test_best_metric_callback_does_not_update_on_worse_metric(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test BestMetricCallback doesn't update best when metric doesn't improve."""
        from noether.core.callbacks.online.best_metric import BestMetricCallback
        from noether.core.schemas.callbacks import BestMetricCallbackConfig

        mock_metric_property_provider.higher_is_better = Mock(return_value=True)
        mock_log_writer.log_cache = {"val/accuracy": 0.8, "test/accuracy": 0.75}

        config = BestMetricCallbackConfig.model_validate(
            dict(
                source_metric_key="val/accuracy",
                target_metric_keys=["test/accuracy"],
                every_n_updates=1,
            )
        )

        callback = BestMetricCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        # Set initial best
        callback.best_metric_value = 0.9
        callback.previous_log_values = {"test/accuracy/at_best/val/accuracy": 0.85}

        callback.periodic_callback(trainer=mock_trainer)

        # Best value should not change
        assert callback.best_metric_value == 0.9

    def test_best_metric_callback_with_optional_metrics(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test BestMetricCallback handles optional target metrics."""
        from noether.core.callbacks.online.best_metric import BestMetricCallback
        from noether.core.schemas.callbacks import BestMetricCallbackConfig

        mock_metric_property_provider.higher_is_better = Mock(return_value=True)
        mock_log_writer.log_cache = {
            "val/accuracy": 0.9,
            "test/accuracy": 0.85,
            "test/f1": 0.88,
        }

        config = BestMetricCallbackConfig.model_validate(
            dict(
                source_metric_key="val/accuracy",
                target_metric_keys=["test/accuracy"],
                optional_target_metric_keys=["test/f1", "test/missing"],
                every_n_updates=1,
            )
        )

        callback = BestMetricCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        callback.periodic_callback(trainer=mock_trainer)

        # Should log both mandatory and available optional metrics
        assert callback.best_metric_value == 0.9
        assert "test/accuracy/at_best/val/accuracy" in callback.previous_log_values
        assert "test/f1/at_best/val/accuracy" in callback.previous_log_values
        # Missing optional metric should not cause issues


class TestTrackAdditionalOutputsCallback:
    """Tests for TrackAdditionalOutputsCallback."""

    def test_update_output_callback_tracks_by_keys(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test TrackAdditionalOutputsCallback tracks outputs by exact key match."""
        from noether.core.callbacks.online.track_outputs import TrackAdditionalOutputsCallback
        from noether.core.schemas.callbacks import TrackAdditionalOutputsCallbackConfig

        config = TrackAdditionalOutputsCallbackConfig.model_validate(
            dict(
                keys=["grad_norm", "param_norm"],
                every_n_updates=10,
                reduce="mean",
                log_output=True,
                save_output=False,
            )
        )
        mock_trainer.update_counter = Mock()
        mock_trainer.update_counter.update = 1

        callback = TrackAdditionalOutputsCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        update_outputs = {
            "grad_norm": torch.tensor(1.5),
            "param_norm": torch.tensor(2.5),
            "other_value": torch.tensor(3.5),
        }

        callback.track_after_accumulation_step(
            update_counter=mock_trainer.update_counter,
            update_outputs=update_outputs,
        )

        assert "grad_norm" in callback.tracked_values
        assert "param_norm" in callback.tracked_values
        assert "other_value" not in callback.tracked_values
        assert len(callback.tracked_values["grad_norm"]) == 1

    def test_update_output_callback_tracks_by_patterns(
        self,
        mock_trainer,
        mock_model,
        mock_data_container,
        mock_tracker,
        mock_log_writer,
        mock_checkpoint_writer,
        mock_metric_property_provider,
    ):
        """Test TrackAdditionalOutputsCallback tracks outputs by pattern matching."""
        from noether.core.callbacks.online.track_outputs import TrackAdditionalOutputsCallback
        from noether.core.schemas.callbacks import TrackAdditionalOutputsCallbackConfig

        config = TrackAdditionalOutputsCallbackConfig.model_validate(
            dict(
                patterns=["loss"],
                every_n_updates=10,
                reduce="mean",
                log_output=True,
                save_output=False,
            )
        )
        mock_trainer.update_counter = Mock()
        mock_trainer.update_counter.update = 1

        callback = TrackAdditionalOutputsCallback(
            callback_config=config,
            trainer=mock_trainer,
            model=mock_model,
            data_container=mock_data_container,
            tracker=mock_tracker,
            log_writer=mock_log_writer,
            checkpoint_writer=mock_checkpoint_writer,
            metric_property_provider=mock_metric_property_provider,
        )

        update_outputs = {
            "total_loss": torch.tensor(0.5),
            "reconstruction_loss": torch.tensor(0.3),
            "grad_norm": torch.tensor(1.5),
        }

        callback.track_after_accumulation_step(
            update_counter=mock_trainer.update_counter,
            update_outputs=update_outputs,
        )

        assert "total_loss" in callback.tracked_values
        assert "reconstruction_loss" in callback.tracked_values
        assert "grad_norm" not in callback.tracked_values
