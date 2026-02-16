#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import os
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_IMPORT_ERROR = None
except ImportError as e:
    TENSORBOARD_IMPORT_ERROR = e
    # Define SummaryWriter as Any to avoid NameErrors if import fails
    SummaryWriter = Any  # type: ignore

from noether.core.schemas.trackers import TensorboardTrackerSchema
from noether.core.trackers.base import BaseTracker


class TensorboardTracker(BaseTracker):
    """TensorBoard tracker for logging metrics and configuration."""

    def __init__(
        self,
        tracker_config: TensorboardTrackerSchema,
        **kwargs,
    ) -> None:
        """Initialize the TensorboardTracker.

        Args:
            tracker_config: Configuration for TensorBoard. See :class:`~noether.core.schemas.trackers.TensorboardTrackerSchema`
                            for available options (typically including base `log_dir`).
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.config = tracker_config
        self.writer: SummaryWriter | None = None
        self._internal_step = 0
        self._config: dict[str, Any] = {}
        self._summary: dict[str, Any] = {}

    def _init(self, config: dict, output_uri: str, run_id: str):
        if TENSORBOARD_IMPORT_ERROR is not None:
            raise ImportError(
                "TensorBoard is not installed. Please install `tensorboard` and `torch` to use the TensorboardTracker. "
                f"Original error: {TENSORBOARD_IMPORT_ERROR}"
            ) from TENSORBOARD_IMPORT_ERROR

        self.logger.info("Initializing TensorBoard")

        name = config.get("run_name", "experiment")
        if "stage_name" in config and config["stage_name"] is not None:
            name = f"{name}/{config['stage_name']}"

        # Replace any forward slashes in the name with an underscore
        name = name.replace("/", "_")

        print(f"Initializing TensorBoard tracker for run: {name} (ID: {run_id})")

        # TensorBoard separates runs by directories.
        base_log_dir = getattr(self.config, "log_dir", "./runs")
        run_log_dir = os.path.join(base_log_dir, f"{name}_{run_id}")

        flush_secs = getattr(self.config, "flush_secs", 60)

        self.writer = SummaryWriter(log_dir=run_log_dir, flush_secs=flush_secs)

        # Store the config so _close() can access it later
        self._config = config.copy()
        self._log_config_as_text()

    def _log(self, data: dict):
        if self.writer is None:
            raise RuntimeError("TensorBoard writer is not initialized.")

        log_data = data.copy()

        # Extract step if provided in the dictionary, otherwise use step internal counter
        step = log_data.pop("step", log_data.pop("global_step", self._internal_step))

        for key, value in log_data.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
            elif isinstance(value, str):
                self.writer.add_text(key, value, step)
            else:
                self.logger.warning(f"Unsupported data type for key '{key}' in TensorBoard tracker: {type(value)}")

        # Increment the internal step counter
        if step >= self._internal_step:
            self._internal_step = step + 1

    def _set_summary(self, key: str, value: Any):
        self._summary[key] = value

    def _update_summary(self, data: dict):
        self._summary.update(data)

    def _log_config_as_text(self):
        """Helper method to log the config dictionary to TensorBoard's text tab."""
        if self.writer:
            markdown_text = "### Run Configuration\n\n"
            for k, v in self._config.items():
                markdown_text += f"* **{k}**: `{v}`\n"
            self.writer.add_text("Configuration", markdown_text, 0)

    def _close(self):
        if self.writer is not None:
            if self._summary:
                markdown_text = "### Final Summary Metrics\n\n"
                for k, v in self._summary.items():
                    markdown_text += f"* **{k}**: `{v}`\n"

                self.writer.add_text("Summary Metrics", markdown_text, self._internal_step)

            self.writer.close()
            self.writer = None
