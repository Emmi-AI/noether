#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch

from noether.core.callbacks.periodic import PeriodicDataIteratorCallback


class BoilerplateCallback(PeriodicDataIteratorCallback):
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
