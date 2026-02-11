#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

try:
    import trackio

    TRACKIO_IMPORT_ERROR = None
except ImportError as e:
    TRACKIO_IMPORT_ERROR = e

from noether.core.schemas.trackers import TrackioTrackerSchema
from noether.core.trackers.base import BaseTracker


class TrackioTracker(BaseTracker):
    """HuggingFace Trackio tracker.

    https://github.com/gradio-app/trackio

    """

    def __init__(
        self,
        tracker_config: TrackioTrackerSchema,
        **kwargs,
    ) -> None:
        """Initialize the TrackioTracker.
        Args:
            tracker_config: Configuration for the TrackioTracker. See :class:`~noether.core.schemas.trackers.TrackioTrackerSchema` for available options including project and space_id.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.config = tracker_config

    def _init(self, config: dict, output_uri: str, run_id: str):
        if TRACKIO_IMPORT_ERROR is not None:
            raise ImportError(
                f"Trackio is not installed. Please install it to use the TrackioTracker. Original error: {TRACKIO_IMPORT_ERROR}"
            ) from TRACKIO_IMPORT_ERROR

        self.logger.info("initializing trackio")
        # restore original argv (can be modified to comply with hydra but allow --hp and --devices)
        name = config["run_name"]
        if "stage_name" in config and config["stage_name"] is not None:
            name = f"{name}/{config['stage_name']}"
        self.run = trackio.init(
            project=self.config.project,
            space_id=self.config.space_id,
            name=f"{name}_{run_id}",
            config=config,
        )

    def _log(self, data: dict):
        trackio.log(data)

    def _set_summary(self, key, value):
        if self.run is None:
            raise RuntimeError("Trackio run is not initialized, cannot set summary.")
        self.run.config[key] = value

    def _update_summary(self, data: dict):
        if self.run is None:
            raise RuntimeError("Trackio run is not initialized, cannot update summary.")
        self.run.config.update(data)

    def _close(self):
        trackio.finish()
