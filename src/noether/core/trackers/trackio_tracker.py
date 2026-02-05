#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

try:
    import trackio
except ImportError as e:
    raise ImportError(
        "trackio is not installed. Please install it with `pip install trackio` to use the TrackioTracker."
    ) from e
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
            tracker_config: Configuration for the TrackioTracker.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.config = tracker_config

    def _init(self, config: dict, output_uri: str, run_id: str):
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
