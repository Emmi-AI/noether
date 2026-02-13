#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class InitializerConfig(BaseModel):
    kind: str = Field(default="noether.core.initializers.PreviousRunInitializer")
    kwargs: dict[str, Any] | None = None
    """Additional keyword arguments to pass to the initializer."""
    run_id: str
    """A unique identifier for the training stage. This is used to find the correct checkpoint."""

    stage_name: str | None = None
    """The name of the stage training stage if defined. When training, the stage name is usually "train"."""
    model_name: str | None = None
    """The name of the model to load. This is the model_name used in CheckpointCallback."""
    checkpoint: str | None | dict = None
    """Which checkpoint to load.
    Checkpoint is usually "latest" or "best_loss", or "E*_U*_S*", depending on which checkpoint you want to load.
    """
    model_info: str | None = None
    """Optional string that is used as suffix in the checkpoint filename."""
    model_config = {"extra": "forbid"}


class CheckpointInitializerConfig(InitializerConfig):
    kind: Literal["noether.core.initializers.CheckpointInitializer"] = Field(
        default="noether.core.initializers.CheckpointInitializer", frozen=True
    )
    load_optim: bool = Field(...)
    """Whether or not to load the optimizer state from the checkpoint. Default is True, as this is usually used to resume a training run"""
    pop_ckpt_kwargs_keys: list[str] | None = Field(None)
    """which checkpoint to load. If a string is provided, must be one of ("latest", "best_loss"). If a dictionary is provided, must contain keys "epoch", "update", "sample" to identify the checkpoint."""


class ResumeInitializerConfig(CheckpointInitializerConfig):
    kind: Literal["noether.core.initializers.ResumeInitializer"] = Field(
        default="noether.core.initializers.ResumeInitializer", frozen=True
    )  # type: ignore[assignment]
    load_optim: bool = Field(True, frozen=True)
    model_name: str = Field(...)


class PreviousRunInitializerConfig(CheckpointInitializerConfig):
    kind: Literal["noether.core.initializers.PreviousRunInitializer"] = Field(
        default="noether.core.initializers.PreviousRunInitializer", frozen=True
    )  # type: ignore[assignment]
    load_optim: bool = Field(False, frozen=True)
    keys_to_remove: list[str] | None = Field(
        None,
    )
    """List of keys to remove from the checkpoint."""
    patterns_to_remove: list[str] | None = Field(None)
    """List of patterns to remove from the checkpoint."""
    patterns_to_rename: list[dict] | None = Field(None)
    """List of patterns to rename in the checkpoint."""
    patterns_to_instantiate: list[str] | None = Field(None)
    """List of patterns to instantiate in the checkpoint."""


AnyInitializer = Union[CheckpointInitializerConfig, ResumeInitializerConfig, PreviousRunInitializerConfig]
