#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from unittest.mock import MagicMock, patch

from noether.core.schemas.initializers import PreviousRunInitializerConfig
from noether.inference.runners.inference_runner import InferenceRunner

_MODULE_PATH = "noether.inference.runners.inference_runner.InferenceRunner"


class TestInferenceRunner:
    @patch(_MODULE_PATH + ".setup_experiment")
    def test_main_execution_flow(self, mock_setup):
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        mock_tracker = MagicMock()
        mock_message_counter = MagicMock()

        mock_setup.return_value = (
            mock_trainer,
            mock_model,
            mock_tracker,
            mock_message_counter,
        )

        mock_config = MagicMock()
        device = "cuda:0"

        InferenceRunner.main(device=device, config=mock_config)

        mock_setup.assert_called_once_with(
            device=device,
            config=mock_config,
            initializer_config_class=PreviousRunInitializerConfig,
        )

        mock_trainer.eval.assert_called_once_with(mock_model)
        mock_tracker.summarize_logvalues.assert_called_once()
        mock_message_counter.log.assert_called_once()
        mock_tracker.close.assert_called_once()
