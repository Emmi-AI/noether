#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from pydantic import BaseModel, Field


class WandBTrackerSchema(BaseModel):
    kind: Literal["noether.core.trackers.WandBTracker"] = Field(
        default="noether.core.trackers.WandBTracker", frozen=True
    )
    entity: str | None = Field(None)
    """The entity name for the W&B project."""
    project: str | None = Field(None)
    """The project name for the W&B project."""
    mode: Literal["disabled", "online", "offline"] | None = Field(default="online")
    """he mode of W&B. Can be 'disabled', 'online', or 'offline'."""


class TrackioTrackerSchema(BaseModel):
    """Schema for TrackioTracker configuration."""

    kind: Literal["noether.core.trackers.TrackioTracker"] = Field(
        default="noether.core.trackers.TrackioTracker", frozen=True
    )

    project: str
    """The project name for the Trackio project."""

    space_id: str | None = Field(None)
    """The HuggingFace space ID where to store the Trackio data."""


class TensorboardTrackerSchema(BaseModel):
    """Schema for TensorboardTracker configuration."""

    kind: Literal["noether.core.trackers.TensorboardTracker"] = Field(
        default="noether.core.trackers.TensorboardTracker", frozen=True
    )

    log_dir: str = Field(default="runs")
    """The base directory where TensorBoard event files will be stored."""

    flush_secs: int = Field(default=60)
    """How often, in seconds, to flush the pending events to disk."""


AnyTracker = WandBTrackerSchema | TrackioTrackerSchema | TensorboardTrackerSchema
