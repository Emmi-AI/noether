#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch

from noether.core.callbacks.periodic import PeriodicDataIteratorCallback
from noether.core.schemas.callbacks import OfflineLossCallbackConfig


class OfflineLossCallback(PeriodicDataIteratorCallback):
    """A periodic Callback that is invoked at the end of each epoch to calculate and track the loss and a dataset."""

    def __init__(self, callback_config: OfflineLossCallbackConfig, **kwargs):
        """

        Args:
            callback_config: configuration of the OfflineLossCallback. See :class:`~noether.core.schemas.callbacks.OfflineLossCallbackConfig` for the available options.
        """
        super().__init__(callback_config=callback_config, **kwargs)
        self.dataset_key = callback_config.dataset_key
        self.output_patterns_to_log = callback_config.output_patterns_to_log or []

    def process_data(self, batch, *, trainer_model, **_):
        train_out = self.trainer.train_step(batch=batch, model=trainer_model)

        # extract
        all_losses = {"total": train_out.total_loss.detach().cpu()}
        if train_out.losses_to_log is not None:
            all_losses.update({name: loss.cpu() for name, loss in train_out.losses_to_log.items()})
        outputs_to_log = {}
        for key, value in (train_out.additional_outputs or {}).items():
            for pattern in self.output_patterns_to_log:
                if pattern in key:
                    if not torch.is_tensor(value):
                        value = torch.tensor(value)
                    outputs_to_log[key] = value.cpu()
        return all_losses, outputs_to_log

    def process_results(self, results: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]], **_) -> None:
        losses, outputs = results

        # log losses
        for loss_name, loss in losses.items():
            if loss.ndim != 1:
                raise ValueError("Loss has to be calculated sample-wise to avoid errors through batch")
            # log loss
            mean_loss = loss.mean()
            self.writer.add_scalar(
                key=f"loss/{self.dataset_key}/{loss_name}",
                value=mean_loss,
                logger=self.logger,
                format_str=".5f",
            )
            # log difference to train loss
            if self.writer.log_cache is None:
                raise ValueError("Log cache is empty, can't process results.")

            train_loss = self.writer.log_cache.get(f"loss/online/{loss_name}/{self.to_short_interval_string()}", None)
            if train_loss is not None:
                self.writer.add_scalar(
                    key=f"lossdiff/{self.dataset_key}/{loss_name}",
                    value=mean_loss - train_loss,
                    logger=self.logger,
                    format_str=".5f",
                )
        # log outputs
        for name, output in outputs.items():
            if output.ndim != 1:
                raise ValueError(f"Output has to be calculated sample-wise (name={name} shape={output.shape})")
            self.writer.add_scalar(
                key=f"{name}/{self.dataset_key}",
                value=output.float().mean(),
                logger=self.logger,
                format_str=".5f",
            )
