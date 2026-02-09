#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import os

import wandb

from noether.core.schemas.trackers import WandBTrackerSchema
from noether.core.trackers.base import BaseTracker


class WandBTracker(BaseTracker):
    """Weights and Biases tracker.

    Example config:

    .. code-block:: yaml
        - kind: noether.core.trackers.WandBTracker
          mode: online
          entity: my_entity
          project: my_project

    """

    MODES = ["disabled", "online", "offline"]

    def __init__(
        self,
        tracker_config: WandBTrackerSchema,
        **kwargs,
    ) -> None:
        """
        Args:
            tracker_config: Configuration for the WandBTracker. See :class:`~noether.core.schemas.trackers.WandBTrackerSchema` for available options including mode, entity, and project.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        assert tracker_config.mode in WandBTracker.MODES
        self.mode = tracker_config.mode
        self.entity = tracker_config.entity
        self.project = tracker_config.project

    def _init(self, config: dict, output_uri: str, run_id: str):
        self.logger.info("------------------")
        self.logger.info(f"initializing wandb (mode={self.mode})")
        if self.mode != "disabled":
            if self.mode == "offline":
                os.environ["WANDB_MODE"] = "offline"
            self.logger.info("logging into wandb")
            wandb.login()
            self.logger.info("logged into wandb")
            # restore original argv (can be modified to comply with hydra but allow --hp and --devices)
            name = config["run_name"]
            if "stage_name" in config and config["stage_name"] is not None:
                name = f"{name}/{config['stage_name']}"
            wandb.init(
                entity=self.entity,
                project=self.project,
                name=name,
                dir=output_uri,
                save_code=False,
                config=config,
                mode=self.mode,
                id=run_id,
                tags=["new"],
            )

    def _log(self, data: dict):
        wandb.log(data)

    def _set_summary(self, key, value):
        if wandb.run is None:
            raise RuntimeError("WandB run is not initialized, cannot set summary.")
        wandb.run.summary[key] = value

    def _update_summary(self, data: dict):
        if wandb.run is None:
            raise RuntimeError("WandB run is not initialized, cannot update summary.")
        wandb.run.summary.update(data)

    def _close(self):
        if self.mode == "disabled":
            return
        wandb.finish()
