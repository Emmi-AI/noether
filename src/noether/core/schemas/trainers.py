#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from noether.core.schemas.callbacks import CallbacksConfig
from noether.core.schemas.initializers import InitializerConfig


class CheckpointConfig(BaseModel):
    epoch: int | None = None
    update: int | None = None
    sample: int | None = None


class BaseTrainerConfig(BaseModel):
    kind: str

    max_epochs: int | None = Field(None)
    """The maximum number of epochs to train for. Mutually exclusive with max_updates and max_samples. If set to 0, training will be skipped and all callbacks will be invoked once (useful for evaluation-only runs)."""
    max_updates: int | None = Field(None)
    """The maximum number of updates to train for. Mutually exclusive with max_epochs and max_samples. If set to 0, training will be skipped and all callbacks will be invoked once (useful for evaluation-only runs)."""
    max_samples: int | None = Field(None)
    """The maximum number of samples to train for. Mutually exclusive with max_epochs and max_updates. If set to 0, training will be skipped and all callbacks will be invoked once (useful for evaluation-only runs)."""

    start_at_epoch: int | None = Field(None)
    """The epoch to start training at. This means that the trainer will skip all epochs before this epoch. Learning rate and other schedulers will be stepped accordingly. Useful for resuming training from a specific epoch."""

    add_default_callbacks: bool | None = Field(True)
    """Whether to add default callbacks. Default callbacks log things like simple dataset statistics or the current value of the learning rate if it is scheduled."""
    add_trainer_callbacks: bool | None = Field(True)
    """Whether to add trainer specific callbacks (e.g., a callback to log the training accuracy for a classification task)."""

    effective_batch_size: int = Field(...)
    """The effective batch size used for optimization. This is the number of samples that are processed before an update step is taken: the "global batch size". In multi-GPU setups, the batch size per device, ("local batch size") is `effective_batch_size / number of devices`. If gradient accumulation is used, the forward-pass batch size is derived by dividing by the number of gradient accumulation steps."""
    precision: Literal["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"] = Field("float32")
    """The precision to use for training (e.g., "float32"). Mixed precision training (e.g., "float16" or "bfloat16") can be used to speed up training and reduce memory usage on supported hardware (e.g., NVIDIA GPUs)."""
    callbacks: list[CallbacksConfig] | None = Field(..., description="List of callback configurations")
    """The callbacks to use for training."""
    initializer: InitializerConfig | None = Field(None)
    """The initializer to use for training. Mainly used for resuming training via ResumeInitializer."""

    log_every_n_epochs: int | None = Field(None, ge=1)
    """The integer number of epochs to periodically log at."""
    log_every_n_updates: int | None = Field(None, ge=1)
    """The integer number of updates to periodically log at."""
    log_every_n_samples: int | None = Field(None, ge=1)
    """The integer number of samples to periodically log at."""
    track_every_n_epochs: int | None = Field(None, ge=1)
    """The integer number of epochs to periodically track metrics at."""
    track_every_n_updates: int | None = Field(None, ge=1)
    """The integer number of updates to periodically track metrics at."""
    track_every_n_samples: int | None = Field(None, ge=1)
    """The integer number of samples to periodically track metrics at."""

    max_batch_size: int | None = Field(None, ge=1)
    """The maximum batch size to use for model forward pass in training. If the effective_batch_size is larger than max_batch_size, gradient accumulation will be used to simulate the larger batch size. For example, if effective_batch_size=8 and max_batch_size=2, 4 gradient accumulation steps will be taken before each optimizer step."""
    skip_nan_loss: bool = Field(False)
    """Whether to skip NaN losses. These can sometimes occur due to unlucky coincidences. If true, NaN losses will be skipped without terminating the training up until 100 NaN losses occurred in a row."""
    skip_nan_loss_max_count: int = Field(100, ge=1)

    disable_gradient_accumulation: bool = Field(True)
    """Whether to disable gradient accumulation. Gradient accumulation is sometimes used to simulate larger batch sizes, but can lead to worse generalization."""

    use_torch_compile: bool = Field(False)
    """Whether to use `torch.compile` to compile the model for faster training."""

    # find_unused_params should not be set to true if it is not needed (to avoid overhead)
    # but sometimes it is required (e.g. when dynamically freezing/unfreezing parameters)
    # when find_unused_params setting static_graph to true can bring speedup
    find_unused_params: bool = Field(False)
    """Sets the `find_unused_parameters` flag of `DistributedDataParallel`."""
    static_graph: bool = Field(False)
    """Sets the `static_graph` flag of `DistributedDataParallel`."""

    forward_properties: list[str] | None = []
    """Properties (i.e., keys from the batch dict) from the input batch that are used as inputs to the model during the forward pass."""
    target_properties: list[str] | None = []
    """Properties (i.e., keys from the batch dict) from the input batch that are used as targets for the model during the forward pass."""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_callback_frequency(self) -> BaseTrainerConfig:
        """
        Ensures that exactly one frequency ('every_n_*') is specified and
        that 'batch_size' is present if 'every_n_samples' is used.
        """
        if not self.add_default_callbacks:
            return self

        frequency_fields = [
            (
                "log",
                [
                    self.log_every_n_epochs,
                    self.log_every_n_updates,
                    self.log_every_n_samples,
                ],
            ),
            (
                "track",
                [
                    self.track_every_n_epochs,
                    self.track_every_n_updates,
                    self.track_every_n_samples,
                ],
            ),
        ]

        for prefix, fields in frequency_fields:
            if sum(field is not None for field in fields) > 1:
                raise ValueError(
                    f"Only one of '{prefix}_every_n_epochs', '{prefix}_every_n_updates', or '{prefix}_every_n_samples' can be specified."
                )

        if self.log_every_n_samples is not None and self.effective_batch_size is None:
            raise ValueError("'effective_batch_size' must be specified when using 'log_every_n_samples'.")
        if self.track_every_n_samples is not None and self.effective_batch_size is None:
            raise ValueError("'effective_batch_size' must be specified when using 'track_every_n_samples'.")

        return self
