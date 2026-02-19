#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import random
import string
from datetime import date
from pathlib import Path


class PathProvider:
    """Provider that defines at which locations things are stored on the disk. The basic layout is that every training
    run is identified by a `stage_name` (e.g., "pretrain" or "finetune") and `run_id`,
    a string that is unique for each training run. All outputs are stored in a directory defined in the configuration
    that is located in `output_path`. The outputs of a single run will be stored in `output_path/stage_name/run_id`.

    Args:
        output_root_path: The base output directory where outputs should be stored (e.g., /save).
        run_id: Unique identifier of the training run.
        stage_name: Optional identifier for the training stage for easier categorization (e.g., "pretrain" or "finetune").
        debug: If `True`, outputs are stored in a "debug" subfolder.
    """

    def __init__(self, output_root_path: Path, run_id: str, stage_name: str | None = None, debug: bool = False):
        self.output_root = output_root_path
        self.stage_name = stage_name
        self.run_id = run_id
        self.debug = debug

    @staticmethod
    def _mkdir(path: Path) -> Path:
        path.mkdir(exist_ok=True, parents=True)
        return path

    def with_run(self, run_id: str | None = None, stage_name: str | None = None) -> PathProvider:
        return PathProvider(
            output_root_path=self.output_root,
            run_id=run_id if run_id is not None else self.run_id,
            stage_name=stage_name,
        )

    @property
    def run_output_path(self) -> Path:
        """Returns the output_path for a given `stage_name` and `run_id`.

        Returns:
            The output path for the current run.
        """
        if self.debug:
            stage_output_path = self.output_root / "debug" / self.run_id
        else:
            stage_output_path = self.output_root / self.run_id

        if self.stage_name is not None:
            stage_output_path = stage_output_path / self.stage_name

        return PathProvider._mkdir(stage_output_path)

    @property
    def logfile_uri(self) -> Path:
        """Returns the URI where the logfile should be stored (the file where stdout messsage are stored)."""
        return self.run_output_path / "log.txt"

    @property
    def checkpoint_path(self) -> Path:
        """Returns the checkpoint path of the current run."""
        return self._mkdir(self.run_output_path / "checkpoints")

    @property
    def _basetracker_path(self) -> Path:
        """Path where to log things for the BaseTracker"""
        return self._mkdir(self.run_output_path / "basetracker")

    @property
    def basetracker_config_uri(self) -> Path:
        """Independent of whether or not (or which) online tracker is used, the log entries are also written to disk.
        This property defines where the config is written to.
        """
        return self._mkdir(self.run_output_path / "basetracker") / "config.yaml"

    @property
    def basetracker_entries_uri(self) -> Path:
        """Independent of whether or not (or which) online tracker is used, the log entries are also written to disk.
        This property defines where the log entries are written to.
        """
        return self._basetracker_path / "entries.th"

    @property
    def basetracker_summary_uri(self) -> Path:
        """Independent of whether or not (or which) online tracker is used, the log entries are also written to disk.
        This property defines where the summary is written to.
        """
        return self._mkdir(self.run_output_path / "basetracker") / "summary.yaml"

    @staticmethod
    def generate_run_id(seed=None) -> str:
        """Generate a random run ID.

        Args:
            seed: Optional seed for reproducibility.

        Returns:
            A random run ID.
        """
        rng = random.Random(seed)
        return date.today().strftime("%Y-%m-%d_") + "".join(rng.choices(string.ascii_lowercase + string.digits, k=5))

    def link(self, ancestor: PathProvider) -> None:
        """Create a symlink from the current run output to the ancestor run output.

        Args:
            ancestor: The ancestor PathProvider to link to.
        """
        # If the target and current run are the same, we don't need to create a link
        if self.output_root == ancestor.output_root and self.run_id == ancestor.run_id:
            return

        link_path = self.run_output_path / "ancestor"
        if link_path.exists() and link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(ancestor.run_output_path)
