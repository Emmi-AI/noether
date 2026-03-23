#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from noether.core.schemas.lib import _RegistryBase


class BaseTrackerConfig(_RegistryBase):
    """Base configuration for experiment trackers. All tracker configs should inherit from this class."""

    _registry: ClassVar[dict[str, type[BaseModel]]] = {}
    _type_field: ClassVar[str] = "kind"
    kind: str | None = None


class WandBTrackerSchema(BaseTrackerConfig):
    entity: str | None = Field(None)
    """The entity name for the W&B project."""
    project: str | None = Field(None)
    """The project name for the W&B project."""
    mode: Literal["disabled", "online", "offline"] | None = Field(default="online")
    """Tracking mode. Can be 'disabled', 'online', or 'offline'."""


class TrackioTrackerSchema(BaseTrackerConfig):
    """Schema for TrackioTracker configuration."""

    project: str
    """The project name for the Trackio project."""

    space_id: str | None = Field(None)
    """The HuggingFace space ID where to store the Trackio data."""


class TensorboardTrackerSchema(BaseTrackerConfig):
    """Schema for TensorboardTracker configuration."""

    log_dir: str = Field(default="runs")
    """The base directory where TensorBoard event files will be stored."""

    flush_secs: int = Field(default=60)
    """How often, in seconds, to flush the pending events to disk."""


AnyTracker = WandBTrackerSchema | TrackioTrackerSchema | TensorboardTrackerSchema
