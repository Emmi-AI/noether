#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from unittest.mock import Mock, PropertyMock, patch

import pytest
import torch

from noether.core.schemas.callbacks import OfflineLossCallbackConfig
from noether.training.callbacks.offline_loss import OfflineLossCallback

_MODULE_LOGGER_PATH = "noether.training.callbacks.offline_loss.OfflineLossCallback.logger"


@pytest.fixture
def mock_trainer():
    trainer = Mock()
    trainer.device = "cpu"
    # Mock update to return (losses, outputs)
    trainer.update = Mock(return_value=({}, {}))
    return trainer


@pytest.fixture
def mock_log_writer():
    writer = Mock()
    writer.add_scalar = Mock()
    writer.log_cache = {}
    return writer


@pytest.fixture
def callback_deps(mock_trainer, mock_log_writer):
    return {
        "trainer": mock_trainer,
        "model": Mock(),
        "data_container": Mock(),
        "tracker": Mock(),
        "log_writer": mock_log_writer,
        "checkpoint_writer": Mock(),
        "metric_property_provider": Mock(),
    }


class TestOfflineLossCallback:
    def test_init(self, callback_deps):
        config = OfflineLossCallbackConfig(
            every_n_epochs=1, dataset_key="validation", output_patterns_to_log=["accuracy"]
        )
        callback = OfflineLossCallback(callback_config=config, **callback_deps)

        assert callback.dataset_key == "validation"
        assert callback.output_patterns_to_log == ["accuracy"]

    def test_process_data_extracts_losses_and_filters_outputs(self, callback_deps):
        config = OfflineLossCallbackConfig(
            every_n_epochs=1,
            dataset_key="val",
            output_patterns_to_log=["pred_mask"],  # Should only keep keys containing this
        )
        callback = OfflineLossCallback(callback_config=config, **callback_deps)

        mock_batch = {"data": torch.randn(2, 2)}
        mock_model = Mock()

        returned_losses = {"loss_a": torch.tensor(1.0), "loss_b": torch.tensor(2.0)}
        returned_outputs = {
            "pred_mask_1": torch.tensor([0.5]),  # matches pattern
            "pred_mask_2": [0.1],  # matches pattern (list, needs conversion)
            "logits": torch.tensor([0.9]),  # does NOT match pattern
        }
        callback_deps["trainer"].update.return_value = (returned_losses, returned_outputs)

        losses, outputs = callback.process_data(mock_batch, trainer_model=mock_model)

        callback_deps["trainer"].update.assert_called_once_with(dist_model=mock_model, batch=mock_batch, training=False)

        assert "loss_a" in losses
        assert "loss_b" in losses
        assert losses["loss_a"] == 1.0

        assert "pred_mask_1" in outputs
        assert "pred_mask_2" in outputs
        assert "logits" not in outputs  # should be filtered out

        assert torch.is_tensor(outputs["pred_mask_2"])
        assert outputs["pred_mask_2"] == torch.tensor([0.1])

    def test_process_results_logs_simple_metrics(self, callback_deps):
        config = OfflineLossCallbackConfig(every_n_epochs=1, dataset_key="val")
        callback = OfflineLossCallback(callback_config=config, **callback_deps)

        with patch(_MODULE_LOGGER_PATH, new_callable=PropertyMock) as mock_logger:
            mock_log_instance = Mock()
            mock_logger.return_value = mock_log_instance

            # Input Results (Sample-wise)
            # 2 samples in batch
            losses = {"mse": torch.tensor([1.0, 3.0])}  # Mean = 2.0
            outputs = {"accuracy": torch.tensor([0.8, 1.0])}  # Mean = 0.9

            callback.process_results((losses, outputs))

            writer = callback_deps["log_writer"]

            writer.add_scalar.assert_any_call(
                key="loss/val/mse",
                value=torch.tensor(2.0),  # Mean
                logger=mock_log_instance,
                format_str=".5f",
            )

            writer.add_scalar.assert_any_call(
                key="accuracy/val",
                value=torch.tensor(0.9),  # Mean
                logger=mock_log_instance,
                format_str=".5f",
            )

    def test_process_results_logs_loss_difference(self, callback_deps):
        # Config: Interval E1 (Every 1 epoch)
        config = OfflineLossCallbackConfig(every_n_epochs=1, dataset_key="val")
        callback = OfflineLossCallback(callback_config=config, **callback_deps)

        # Setup Log Cache to simulate existing training loss
        # Key format expected: loss/online/{loss_name}/{interval_string}
        # Interval string for every_n_epochs=1 is "E1"
        writer = callback_deps["log_writer"]
        writer.log_cache = {
            "loss/online/mse/E1": 1.5,
        }

        with patch(_MODULE_LOGGER_PATH, new_callable=PropertyMock) as mock_logger:
            mock_log_instance = Mock()
            mock_logger.return_value = mock_log_instance

            losses = {"mse": torch.tensor([2.0, 2.0])}  # Mean = 2.0

            callback.process_results((losses, {}))

            # Diff = Current(2.0) - Train(1.5) = 0.5
            writer.add_scalar.assert_any_call(
                key="lossdiff/val/mse",
                value=torch.tensor(0.5),
                logger=mock_log_instance,
                format_str=".5f",
            )

    def test_process_results_validation_errors(self, callback_deps):
        config = OfflineLossCallbackConfig(every_n_epochs=1, dataset_key="val")
        callback = OfflineLossCallback(callback_config=config, **callback_deps)

        bad_loss = {"mse": torch.tensor([[1.0], [2.0]])}  # 2D
        with pytest.raises(ValueError, match="Loss has to be calculated sample-wise"):
            callback.process_results((bad_loss, {}))

        bad_output = {"acc": torch.tensor(1.0)}  # 0D scalar
        with pytest.raises(ValueError, match="Output has to be calculated sample-wise"):
            callback.process_results(({}, bad_output))

        callback_deps["log_writer"].log_cache = None
        valid_loss = {"mse": torch.tensor([1.0])}
        with pytest.raises(ValueError, match="Log cache is empty"):
            callback.process_results((valid_loss, {}))

    def test_process_data_handles_none_outputs(self, callback_deps):
        config = OfflineLossCallbackConfig(every_n_epochs=1, dataset_key="val")
        callback = OfflineLossCallback(callback_config=config, **callback_deps)

        callback_deps["trainer"].update.return_value = ({"l1": torch.tensor(1.0)}, None)

        losses, outputs = callback.process_data({}, trainer_model=Mock())

        assert losses["l1"] == 1.0
        assert outputs == {}
