#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from noether.core.distributed import is_rank0
from noether.core.providers import PathProvider
from noether.core.trackers import BaseTracker
from noether.core.utils.training import UpdateCounter


class LogWriter:
    """Writes logs into a local file and (optionally) to an online webinterface (identified via the `tracker`).
    All logs that should be written for a certain update in the training will be cached and written to the `tracker`
    all at once. For writing the logs to disk, everything for the full training process is cached and only after
    training finished, the logs are written to disk (writing repeatedly to disk takes a long time).

    Args:
        path_provider: Provides the path to store all logs to the disk after training.
        update_counter: Provides the current training progress add the current epoch/update/sample to every log entry.
            This allows, e.g., changing the x-axis to "epoch" in online visiualization tools.
        tracker: Provides an interface for logging to an online experiment tracking platform.
    """

    def __init__(self, path_provider: PathProvider, update_counter: UpdateCounter, tracker: BaseTracker):
        self.logger = logging.getLogger(type(self).__name__)
        self.path_provider = path_provider
        self.update_counter = update_counter
        self.tracker = tracker
        self.log_entries: list[dict[str, Any]] = []
        self.log_cache: dict[str, Any] | None = None
        self.non_scalar_keys: set[str] = set()

    def get_all_metric_values(self, key: str) -> list[float]:
        """Retrieves all values of a metric from all log entries. Used mainly for integration tests.

        Args:
            key: Identifier of the metric.

        Returns:
            The values of the metric over the course of training. Log entries that do not contain the key are skipped.
        """
        return [entry[key] for entry in self.log_entries if key in entry]

    def finish(self) -> None:
        """Stores all logs from the training to the disk."""
        if len(self.log_entries) == 0 or not is_rank0():
            return
        entries_uri = self.path_provider.basetracker_entries_uri
        self.logger.info(f"writing {len(self.log_entries)} log entries to {entries_uri}")
        # convert into {<key>: {<update0>: <value0>, <update1>: <value1>}}
        result: defaultdict[str, dict[int, Any]] = defaultdict(dict)
        for entry in self.log_entries:
            # update is used instead of the _step item that is commonly used in ML experiment trackers
            update = entry["update"]
            for key, value in entry.items():
                if key == "update":
                    continue
                result[key][update] = value
        torch.save(dict(result), entries_uri)
        # yaml is quite inefficient to store large data quantities
        # with open(entries_uri, "w") as f:
        #     yaml.safe_dump(dict(result), f)

    def _log(self, key, value, is_scalar, logger=None, format_str=None):
        key = key.removesuffix("/")
        if self.log_cache is None:
            self.log_cache = dict(
                epoch=self.update_counter.epoch,
                update=self.update_counter.update,
                sample=self.update_counter.sample,
            )
        if key in self.log_cache:
            raise KeyError(f"key {key!r} was already logged for update {self.update_counter.update}")
        self.log_cache[key] = value
        if logger is not None:
            assert is_scalar
            if format_str is not None:
                value = f"{value:{format_str}}"
            logger.info(f"{key}: {value}")
        if not is_scalar:
            self.non_scalar_keys.add(key)

    def flush(self) -> None:
        """Composes a log entry with all metrics that were calculated for the current training state. This method is
        called after every update and if no metrics were logged, it simply does nothing.
        """
        if self.log_cache is None:
            return
        self.tracker.log(self.log_cache)
        # wandb doesn't support querying offline logfiles so offline mode would have no way to summarize stages
        # also fetching the summaries from the online version potentially takes a long time, occupying GPU servers
        # for primitive tasks
        # -------------------
        # wandb has weird behavior when lots of logs are done separately -> collect all log values and log once
        # -------------------
        # check that every log is fully cached (i.e. no update is logged twice)
        if len(self.log_entries) > 0:
            assert self.log_cache["update"] > self.log_entries[-1]["update"]
        # filter out non-scalar keys for summary
        scalar_entries = {key: value for key, value in self.log_cache.items() if key not in self.non_scalar_keys}
        # keep history for summary
        self.log_entries.append(scalar_entries)
        self.log_cache = None
        self.non_scalar_keys.clear()

    def add_scalar(
        self,
        key: str,
        value: torch.Tensor | np.generic | float,
        logger: logging.Logger | None = None,
        format_str: str | None = None,
    ) -> None:
        """Adds a scalar value to the log.
        Args:
            key: Metric identifier.
            value: Scalar tensor or float with the value that should be logged.
            logger: If defined, will log the value to stdout.
            format_str: If defined, will alter the log to stdout to be in the provided format.
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        # scalars are stored with torch.save for summary and loaded again with torch.load
        # torch.load changed default behavior to weights_only=True
        # convert numpy values to primitive types
        if isinstance(value, np.generic):
            assert np.isscalar(value)
            value = value.item()  # type: ignore
        self._log(key, value, logger=logger, format_str=format_str, is_scalar=True)

    def add_nonscalar(self, key: str, value: Any) -> None:
        """Adds a non-scalar value to the log.
        Args:
            key: Metric identifier.
            value: Non-scalar value that should be logged (e.g., wandb.Image, wandb.Histogram, ...).
        """
        self._log(key, value, is_scalar=False)
