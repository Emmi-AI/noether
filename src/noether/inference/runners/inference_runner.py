#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging

from noether.core.schemas.initializers import PreviousRunInitializerConfig
from noether.core.schemas.schema import ConfigSchema
from noether.training.runners.hydra_runner import HydraRunner

logger = logging.getLogger(__name__)


class InferenceRunner(HydraRunner):
    """Runs an inference experiment using @hydra.main as entry point."""

    @staticmethod
    def main(device: str, config: ConfigSchema) -> None:
        """Main method for inference."""
        trainer, model, tracker, message_counter = InferenceRunner.setup_experiment(
            device=device,
            config=config,
            initializer_config_class=PreviousRunInitializerConfig,
        )

        trainer.eval(model)
        tracker.summarize_logvalues()
        message_counter.log()
        tracker.close()
