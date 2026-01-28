#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from noether.core.callbacks.periodic import PeriodicCallback
from noether.core.distributed import all_gather_nograd, all_reduce_mean_grad
from noether.core.schemas.callbacks import TrackAdditionalOutputsCallbackConfig
from noether.core.utils.training import UpdateCounter  # fixme?


class TrackAdditionalOutputsCallback(PeriodicCallback):
    """Callback that is invoked during training after every gradient step to track certain outputs from the update step.
    The update_outputs that are provided in the track_after_accumulation_step method are the additional_outputs field from the TrainerResult returned by the trainer's update step.


    The provided `update_outputs` are assumed to be a dictionary and outputs that match keys or patterns are tracked.
    An update output matches if either the key matches exactly, e.g. {"some_output": ...} and keys["some_output"];
    or if one of the patterns is contained in the update key name, e.g. {"some_loss": ...} and patterns = ["loss"].
    Example config:
        .. code-block:: yaml
            - kind: noether.core.callbacks.TrackAdditionalOutputsCallback
              name: TrackAdditionalOutputsCallback
              every_n_updates: 1
              keys:
                - "surface_pressure_loss"
    """

    out: Path | None

    def __init__(
        self,
        callback_config: TrackAdditionalOutputsCallbackConfig,
        **kwargs,
    ):
        """Initializes the TrackAdditionalOutputsCallback class.

        Args:
            callback_config: The configuration for the callback.
            **kwargs: additional keyword arguments provided to the parent cla
        """

        super().__init__(callback_config=callback_config, **kwargs)

        if not (
            callback_config.keys is None
            or (isinstance(callback_config.keys, list) and all(isinstance(k, str) for k in callback_config.keys))
        ):
            raise ValueError("keys must be a list of strings or None.")

        if not (
            callback_config.patterns is None
            or (
                isinstance(callback_config.patterns, list) and all(isinstance(p, str) for p in callback_config.patterns)
            )
        ):
            raise ValueError("patterns must be a list of strings or None.")

        self.patterns = callback_config.patterns or []
        self.keys = callback_config.keys or []
        if not (len(self.keys) > 0 or len(self.patterns) > 0):
            raise ValueError("Either keys or patterns have to be provided.")
        self.verbose = callback_config.verbose
        self.tracked_values: defaultdict[str, list] = defaultdict(list)
        self.reduce = callback_config.reduce
        self.log_output = callback_config.log_output
        self.save_output = callback_config.save_output
        if self.save_output:
            self.out = self.checkpoint_writer.path_provider.run_output_path / "update_outputs"
            self.out.mkdir(exist_ok=True)
        else:
            self.out = None

    def __str__(self):
        return (
            f"{type(self).__name__}({self.get_interval_string_verbose()}, keys={self.keys}, patterns={self.patterns})"
        )

    def track_after_accumulation_step(self, *, update_counter, update_outputs, **_) -> None:
        """
        Tracks the specified outputs after each accumulation step.

        :param update_counter: UpdateCounter object to track the number of updates.
        :param update_outputs: The additional_outputs field from the TrainerResult returned by the trainer's update step. Note that the base train_step method in the base trainer does not provide any additional outputs by default, and hence this callback can only be used if the train_step is modified to provide additional outputs.

        """
        if update_outputs is None:
            return

        if self.reduce == "last" and self.updates_till_next_log(update_counter) > 1:
            return
        if len(self.keys) > 0:
            for key in self.keys:
                value = update_outputs[key]
                if torch.is_tensor(value):
                    value = value.detach()
                self.tracked_values[key].append(value)
        if len(self.patterns) > 0:
            for key, value in update_outputs.items():
                for pattern in self.patterns:
                    if pattern in key:
                        value = update_outputs[key]
                        if torch.is_tensor(value):
                            value = value.detach()
                        self.tracked_values[key].append(value)

    def periodic_callback(self, *, update_counter: UpdateCounter, **_) -> None:
        for key, tracked_values in self.tracked_values.items():
            if self.reduce == "mean":
                if torch.is_tensor(tracked_values[0]):
                    reduced_value: torch.Tensor = torch.stack(tracked_values).float().mean()
                else:
                    reduced_value = torch.tensor(np.mean(tracked_values))
                reduced_value = all_reduce_mean_grad(reduced_value)
            elif self.reduce == "last":
                assert len(tracked_values) == 1
                reduced_value = all_gather_nograd(tracked_values[0])
            else:
                raise NotImplementedError
            if self.log_output:
                assert reduced_value.numel() == 1
                self.writer.add_scalar(
                    key=f"{key}/{self.to_short_interval_string()}",
                    value=reduced_value,
                    logger=self.logger if self.verbose else None,
                    format_str=".5f",
                )
            if self.save_output and self.out is not None:
                uri = self.out / f"{key}_{self.to_short_interval_string()}_{update_counter.cur_iteration}.th"
                torch.save(reduced_value, uri)
        self.tracked_values.clear()
