#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
from pathlib import Path
from typing import Any


import yaml
from pydantic import BaseModel

from noether.core.schemas.schema import ConfigSchema

_logger = logging.getLogger(__name__)

def _collect_computed_field_names() -> set[str]:
    """Return all computed_field names across every loaded pydantic BaseModel subclass."""
    names: set[str] = set()
    visited: set[type] = set()
    to_visit: list[type] = list(BaseModel.__subclasses__())
    while to_visit:
        cls = to_visit.pop()
        if cls in visited:
            continue
        visited.add(cls)
        to_visit.extend(cls.__subclasses__())
        names.update(cls.model_computed_fields.keys())
    return names


def strip_computed_fields(data: Any) -> Any:
    """Recursively drop any dict key matching a computed_field name in the loaded schema tree.

    Computed fields are re-derived at validation time; they must not appear as input to
    pydantic models configured with `extra='forbid'`, which would otherwise reject them.
    """
    names = _collect_computed_field_names()

    def _walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items() if k not in names}
        if isinstance(obj, list):
            return [_walk(x) for x in obj]
        return obj

    return _walk(data)

class Hyperparameters:
    """Utility class to store and log hyperparameters configurations from a Pydantic model."""

    @staticmethod
    def save_resolved(stage_hyperparameters: ConfigSchema, out_file_uri: str | Path) -> None:
        """Save the resolved config schema hyperparameters to the output file.


        Args:
            stage_hyperparameters: Hyperparameters to save in a Pydantic object.
            out_file_uri: Path to the output file.
        Returns:
            None
        """

        with open(out_file_uri, "w") as f:
            config_dict = stage_hyperparameters.model_dump(exclude_unset=True)
            config_dict = strip_computed_fields(config_dict)
            config_dict["config_schema_kind"] = stage_hyperparameters.config_schema_kind
            yaml.dump(config_dict, f, sort_keys=False)

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
