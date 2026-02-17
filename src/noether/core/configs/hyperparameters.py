#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel

from noether.core.schemas.schema import ConfigSchema

_logger = logging.getLogger(__name__)


class Hyperparameters:
    """Utility class to store and log hyperparameters configurations from a Pydantic model."""

    @staticmethod
    def save_resolved(
        stage_hyperparameters: ConfigSchema, out_file_uri: str | Path, exclude_unset: bool = True
    ) -> None:
        """Save the resolved config schema hyperparameters to the output file.


        Args:
            stage_hyperparameters: Hyperparameters to save in a Pydantic object.
            out_file_uri: Path to the output file.

        Returns:
            None
        """

        with open(out_file_uri, "w") as f:
            config_dict = stage_hyperparameters.model_dump(exclude_unset=exclude_unset)
            config_dict["config_schema_kind"] = stage_hyperparameters.config_schema_kind
            yaml.dump(config_dict, f)

        _logger.info(f"Dumped resolved hyperparameters to {out_file_uri}")

    @staticmethod
    def log(stage_hyperparameters: BaseModel) -> None:
        """Logs the stage hyperparameters in YAML format without trailing newlines.

        Args:
            stage_hyperparameters: The hyperparameters configuration to log.

        Returns:
            None
        """
        yaml_str = yaml.dump(stage_hyperparameters.model_dump()).rstrip("\n")
        _logger.debug(yaml_str)
