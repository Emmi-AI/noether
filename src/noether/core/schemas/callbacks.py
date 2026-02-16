#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class CallBackBaseConfig(BaseModel):
    if TYPE_CHECKING:
        name: str
    else:
        name: Literal["BaseCallbacksConfig"] = Field(default="BaseCallbacksConfig", frozen=True)

    kind: str | None = None
    every_n_epochs: int | None = Field(None, ge=0)
    """Epoch-based interval. Invokes the callback after every n epochs. Mutually exclusive with other intervals."""
    every_n_updates: int | None = Field(None, ge=0)
    """Update-based interval. Invokes the callback after every n updates. Mutually exclusive with other intervals."""
    every_n_samples: int | None = Field(None, ge=0)
    """Sample-based interval. Invokes the callback after every n samples. Mutually exclusive with other intervals."""
    batch_size: int | None = None
    """Batch size to use for this callback. Default: None (use the same batch_size as for training)."""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_callback_frequency(self) -> CallBackBaseConfig:
        """
        Ensures that exactly one frequency ('every_n_*') is specified and
        that 'batch_size' is present if 'every_n_samples' is used.
        """

        # 1. Mutual Exclusivity and Presence Validation
        frequency_fields = [self.every_n_epochs, self.every_n_updates, self.every_n_samples]
        num_frequency_fields_set = sum(1 for f in frequency_fields if f is not None)

        if num_frequency_fields_set != 1:
            raise ValueError(
                "Exactly one of 'every_n_epochs', 'every_n_updates', or 'every_n_samples' must be set. Cannot have multiple or none set."
            )

        # 2. Conditional Requirement Validation
        if self.every_n_samples is not None and self.batch_size is None:
            raise ValueError("'batch_size' is required when 'every_n_samples' is set.")

        return self

    @field_validator("every_n_epochs", "every_n_updates", "every_n_samples", "batch_size")
    @classmethod
    def check_positive_values(cls, v: int | None) -> int | None:
        """
        Ensures that all integer-based frequency and batch size fields are positive.
        """
        # 3. Value Constraints
        if v is not None and v <= 0:
            raise ValueError(f"Value must be a positive integer, but got {v}")
        return v

    @field_validator("kind")
    @classmethod
    def check_kind_is_not_empty(cls, v: str) -> str:
        """
        Ensures the 'kind' field is a non-empty string.
        """
        # 4. Field Constraints
        if v is not None and not v.strip():
            raise ValueError("'kind' cannot be an empty string.")
        return v


class PeriodicDataIteratorCallbackConfig(CallBackBaseConfig):
    if TYPE_CHECKING:
        name: str
    else:
        name: Literal["PeriodicDataIteratorCallback"] = Field(default="PeriodicDataIteratorCallback", frozen=True)
    dataset_key: str = Field(...)
    """The key of the dataset to be used for the loss calculation. Can be any key that is registered in the `DataContainer`."""


class BestCheckpointCallbackConfig(CallBackBaseConfig):
    name: Literal["BestCheckpointCallback"] = Field("BestCheckpointCallback", frozen=True)
    metric_key: str = Field(...)
    """"The key of the metric to be used for checking the best model."""
    save_frozen_weights: bool = Field(True)
    """Whether to also save the frozen weights of the model."""
    tolerances: list[int] | None = Field(
        None,
    )
    """"If provided, this callback will produce multiple best models which differ in the amount of intervals they allow the metric to not improve. For example, tolerance=[5] with every_n_epochs=1 will store a checkpoint where at most 5 epochs have passed until the metric improved. Additionally, the best checkpoint over the whole training will always be stored (i.e., tolerance=infinite). When setting different tolerances, one can evaluate different early stopping configurations with one training run."""
    model_names: list[str] | None = Field(None)
    """Which model name to save (e.g., if only the encoder of an autoencoder should be stored, one could pass model_name='encoder' here). This only applies when training a CompositeModel. If None, all models are saved."""


class CheckpointCallbackConfig(CallBackBaseConfig):
    name: Literal["CheckpointCallback"] = Field("CheckpointCallback", frozen=True)

    save_weights: bool = Field(True)
    """Whether to save the weights of the model every time this callback is invoked. The checkpoint name will contain the training iteration (e.g., epoch/update/sample) at which the checkpoint was saved."""
    save_optim: bool = Field(False)
    """Whether to save the optimizer state every time this callback is invoked. The checkpoint name will contain the training iteration (e.g., epoch/update/sample) at which the checkpoint was saved."""
    save_latest_weights: bool = Field(True)
    """Whether to save the latest weights of the model every time this callback is invoked. Note that the latest weights are always overwritten on the next invocation of this callback."""
    save_latest_optim: bool = Field(False)
    """Whether to save the latest optimizer state every time this callback is invoked. Note that the latest optimizer state is always overwritten on the next invocation of this callback"""
    model_names: list[str] | None = Field(None)
    """The name of the model to save. If None, all models are saved."""


class EmaCallbackConfig(CallBackBaseConfig):
    name: Literal["EmaCallback"] = Field("EmaCallback", frozen=True)
    target_factors: list[float] = Field(...)
    """The factors for the EMA."""
    model_paths: list[str | None] | None = Field(None)
    """The paths to the models to apply the EMA to (i.e., composite_model.encoder/composite_model.decoder, path of the PyTorch nn.Modules in the checkpoint). If None, the EMA is applied to the whole model. When training with a CompositeModel, the paths on the submodules (i.e., 'encoder', 'decoder', etc.) should be provided via this field, otherwise the EMA will be applied to the CompositeModel as a whole which is not possible to restore later on."""
    save_weights: bool = Field(True)
    """Whether to save the EMA weights."""
    save_last_weights: bool = Field(True)
    """Save the weights of the model when training is over (i.e., at the end of training, save the EMA weights)."""
    save_latest_weights: bool = Field(False)
    """Save the latest EMA weights. Note that the latest weights are always overwritten on the next invocation of this callback."""


class OnlineLossCallbackConfig(CallBackBaseConfig):
    name: Literal["OnlineLossCallback"] = Field("OnlineLossCallback", frozen=True)
    verbose: bool = Field(True)
    """Whether to log the loss."""


class BestMetricCallbackConfig(CallBackBaseConfig):
    name: Literal["BestMetricCallback"] = Field("BestMetricCallback", frozen=True)
    """The metric to use to dermine whether the current model obtained a new best (e.g., loss/valid/total)"""
    source_metric_key: str = Field(...)
    """The metrics to keep track of (e.g., loss/test/total)"""
    target_metric_keys: list[str] | None = Field(None)
    """The metrics to keep track of if they are present (useful when different model configurations log different evaluation metrics to avoid reconfiguring the callback)."""
    optional_target_metric_keys: list[str] | None = Field(None)


class TrackAdditionalOutputsCallbackConfig(CallBackBaseConfig):
    name: Literal["TrackAdditionalOutputsCallback"] = Field("TrackAdditionalOutputsCallback", frozen=True)
    keys: list[str] | None = Field(None)
    """List of keys to track in the additional_outputs of the TrainerResult returned by the trainer's update step."""
    patterns: list[str] | None = Field(None)
    """List of patterns to track in the additional_outputs of the TrainerResult returned by the trainer's update step. Matched if it is contained in one of the update_outputs keys."""
    verbose: bool = Field(False)
    """If True uses the logger to print the tracked values otherwise uses no logger."""
    reduce: Literal["mean", "last"] = Field("mean")
    """The reduction method to be applied to the tracked values to reduce to scalar. Currently supports 'mean' and 'last'."""
    log_output: bool = Field(True)
    """Whether to log the tracked scalar values."""
    save_output: bool = Field(False)
    """Whether to save the tracked scalar values to disk."""


class OfflineLossCallbackConfig(PeriodicDataIteratorCallbackConfig):
    name: Literal["OfflineLossCallback"] = Field("OfflineLossCallback", frozen=True)

    output_patterns_to_log: list[str] | None = Field(None)
    """For instance, if the output key is 'some_loss' and the pattern is ['loss'].  **kwargs: additional arguments passed to the parent class."""


class MetricEarlyStopperConfig(CallBackBaseConfig):
    name: Literal["MetricEarlyStopper"] = Field("MetricEarlyStopper", frozen=True)
    metric_key: str
    """The key of the metric to monitor"""
    tolerance: int
    """The number of times the metric can stagnate before stopping training"""

    @field_validator("tolerance")
    @classmethod
    def check_tolerance_positive(cls, v: int) -> int:
        """
        Ensures that tolerance is at least 1.
        """
        if v < 1:
            raise ValueError(f"'tolerance' must be >= 1, but got {v}")
        return v


class FixedEarlyStopperConfig(BaseModel):
    kind: str | None = None
    name: Literal["FixedEarlyStopper"] = Field("FixedEarlyStopper", frozen=True)
    stop_at_sample: int | None = None
    stop_at_update: int | None = None
    stop_at_epoch: int | None = None

    @model_validator(mode="after")
    def validate_callback_frequency(self) -> FixedEarlyStopperConfig:
        """
        Ensures that exactly one stop ('stop_at_*') is specified
        """
        # 1. Mutual Exclusivity and Presence Validation
        frequency_fields = [self.stop_at_epoch, self.stop_at_update, self.stop_at_sample]
        num_frequency_fields_set = sum(1 for f in frequency_fields if f is not None)

        if num_frequency_fields_set != 1:
            raise ValueError(
                "Exactly one of 'stop_at_epoch', 'stop_at_update', or 'stop_at_sample' must be set. Cannot have multiple or none set."
            )
        return self


CallbacksConfig = Union[
    CallBackBaseConfig
    | BestCheckpointCallbackConfig
    | CheckpointCallbackConfig
    | EmaCallbackConfig
    | OnlineLossCallbackConfig
    | BestMetricCallbackConfig
    | TrackAdditionalOutputsCallbackConfig
    | OfflineLossCallbackConfig
    | MetricEarlyStopperConfig
    | FixedEarlyStopperConfig
    | PeriodicDataIteratorCallbackConfig
]
