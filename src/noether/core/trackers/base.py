#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import os
import platform
from copy import deepcopy
from typing import Any

import torch
import yaml

from noether.core.distributed import get_num_nodes, get_world_size, is_rank0
from noether.core.providers import MetricPropertyProvider, PathProvider


class BaseTracker:
    """Base class for all experiment trackers."""

    def __init__(self, metric_property_provider: MetricPropertyProvider, path_provider: PathProvider):
        """Initialize the BaseTracker.

        Args:
            metric_property_provider: The metric property provider gives additional information such
                as whether higher values are better.
            path_provider: Gives access to paths (e.g., output_path: where checkpoints/logs are stored).
        """
        self.logger = logging.getLogger(type(self).__name__)
        self.metric_property_provider = metric_property_provider
        self.path_provider = path_provider
        self.summary: dict[str, Any] = {}

    def init(
        self,
        accelerator: str,
        run_name: str,
        stage_name: str | None,
        stage_hp: dict,
        run_id: str,
        output_uri: str,
    ) -> None:
        """Initialize the tracker for a specific run.

        Args:
            accelerator: The accelerator used for training (e.g., "cpu", "cuda").
            run_name: The name of the run.
            stage_name: The stage of the run.
            stage_hp: The hyperparameters for the run stage.
            run_id: The ID of the run.
            output_uri: The URI where the output will be stored.
        """
        if not is_rank0():
            return
        if not isinstance(output_uri, str):
            output_uri = output_uri.as_posix()
        config = dict(
            run_name=run_name,
            stage_name=stage_name or "default",
            hp=BaseTracker._lists_to_dict(stage_hp),
        )
        # log additional environment properties
        if accelerator in ["cpu", "mps"]:
            config["accelerator"] = accelerator
        elif accelerator == "gpu":
            config["accelerator"] = torch.cuda.get_device_name(0)
        config["dist/world_size"] = str(get_world_size())
        config["dist/nodes"] = str(get_num_nodes())
        config["dist/hostname"] = platform.uname().node
        if "SLURM_JOB_ID" in os.environ:
            config["dist/jobid"] = os.environ["SLURM_JOB_ID"]
        # save to disk
        with open(self.path_provider.basetracker_config_uri, "w") as f:
            yaml.dump(dict(config), f)
        # init implementing tracker
        self._init(config=config, output_uri=output_uri, run_id=run_id)

    def _init(self, config: dict, output_uri: str, run_id: str):
        raise NotImplementedError

    def log(self, data: dict[str, Any]) -> None:
        """Log data to the tracker.

        Args:
            data: The data to log. This should be a dictionary where the values are the data to log.
        """
        if not is_rank0():
            return
        self._log(data=data)

    def _log(self, data: dict):
        raise NotImplementedError

    def close(self) -> None:
        """Close the file used by the tracker and save the summary."""
        if not is_rank0():
            return
        # store summary on disk
        with open(self.path_provider.basetracker_summary_uri, "w") as f:
            yaml.safe_dump(self.summary, f)
        self._close()

    def _close(self):
        raise NotImplementedError

    def set_summary(self, key: str, value: Any) -> None:
        """Set a summary value.
        Args:
            key: The key for the summary value.
            value: The value to set.
        """
        if not is_rank0():
            return
        self.summary[key] = value
        self._set_summary(key=key, value=value)

    def _set_summary(self, key, value):
        raise NotImplementedError

    def update_summary(self, data: dict):
        """Update the summary with new data.

        Args:
            data: The data to update the summary with. This should be a dictionary where the values are the data to log.
        """
        if not is_rank0():
            return
        self.summary.update(data)
        self._update_summary(data=data)

    def _update_summary(self, data: dict):
        raise NotImplementedError

    def summarize_logvalues(self) -> None:
        """Summarize the log values from the entries.
        This method is called after the training is finished and summarizes the log values.
        It computes the min/max values for each log value and stores them in the summary."""
        entries_uri = self.path_provider.basetracker_entries_uri
        if not entries_uri.exists():
            return None
        entries = torch.load(entries_uri, weights_only=True)
        if entries is None:
            return None
        summary = {}
        for key, update_to_value in entries.items():
            # exclude neutral keys (e.g. lr, profiling, ...) for min/max summarizing
            if self.metric_property_provider.is_neutral_key(key):
                continue

            if key in ["epoch", "update", "sample"]:
                continue
            values = list(update_to_value.values())
            # min/max
            higher_is_better = self.metric_property_provider.higher_is_better(key)
            if higher_is_better:
                minmax_key = f"{key}/max"
                minmax_value = max(values)
            else:
                minmax_key = f"{key}/min"
                minmax_value = min(values)
            summary[minmax_key] = minmax_value
            self.logger.info(f"{minmax_key}: {minmax_value}")
            # add last (wandb adds it automatically, but with the postfix /last it is easier to distinguish)
            last_key = f"{key}/last"
            last_value = values[-1]
            summary[last_key] = last_value

        # cached update
        self.update_summary(summary)

    @staticmethod
    def _lists_to_dict(root):
        """wandb cant handle lists in configs -> transform lists into dicts with str(i) as key"""
        #  (it will be displayed as [{"kind": "..."}, ...])
        root = deepcopy(root)
        return BaseTracker._lists_to_dicts_impl(dict(root=root))["root"]

    @staticmethod
    def _lists_to_dicts_impl(root):
        if not isinstance(root, dict):
            return
        for k, v in root.items():
            if isinstance(v, list):
                root[k] = {str(i): vitem for i, vitem in enumerate(v)}
            elif isinstance(v, dict):
                root[k] = BaseTracker._lists_to_dicts_impl(root[k])
        return root
