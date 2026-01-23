#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch

from boilerplate_project.schemas.callbacks.base_callback_config import BoilerplateCallbackConfig
from noether.core.callbacks.periodic import PeriodicDataIteratorCallback


class BoilerplateCallback(PeriodicDataIteratorCallback):
    def __init__(self, callback_config: BoilerplateCallbackConfig, **kwargs):
        super().__init__(callback_config=callback_config, **kwargs)

        self.dataset_key = callback_config.dataset_key

    def process_data(self, batch: dict[str, torch.Tensor], **_) -> dict[str, torch.Tensor]:
        with self.trainer.autocast_context:
            x = batch["x"]
            model_outputs = self.model(x)

        return {"y_hat": model_outputs, "target": batch["y"].clone()}

    def process_results(self, results, **_) -> None:
        accuracy = (results["y_hat"].argmax(dim=1) == results["target"]).float().mean().item()
        self.writer.add_scalar(
            key=f"metrics/{self.dataset_key}/accuracy",
            value=accuracy,
            logger=self.logger,
            format_str=".6f",
        )
