#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from noether.core.schemas.lib import _RegistryBase
from noether.core.schemas.schema import ConfigSchema

_logger = logging.getLogger(__name__)


def _inject_discriminator_fields(model: Any, dumped: Any) -> None:
    """Re-add discriminator fields (e.g. ``kind``) onto ``dumped`` for every
    ``_RegistryBase`` instance in ``model`` whose discriminator was stripped
    by ``model_dump(exclude_unset=True)``.

    Discriminator fields default to a literal class value, so they're "unset"
    when constructed in Python without an explicit ``kind=...`` argument and
    therefore disappear from the dump — but they're required to re-validate
    the discriminated union on reload (e.g. when ``noether-eval`` reads
    ``hp_resolved.yaml`` back).

    Mutates ``dumped`` in place; recurses through nested models, lists, and
    dicts so the entire tree is patched.
    """
    if isinstance(model, _RegistryBase) and isinstance(dumped, dict):
        type_field = type(model)._type_field
        if type_field not in dumped:
            value = getattr(model, type_field, None)
            if value is not None:
                dumped[type_field] = value

    if isinstance(model, BaseModel) and isinstance(dumped, dict):
        for field_name in type(model).model_fields:
            if field_name in dumped:
                _inject_discriminator_fields(getattr(model, field_name), dumped[field_name])
    elif isinstance(model, list) and isinstance(dumped, list):
        for child_model, child_dumped in zip(model, dumped, strict=False):
            _inject_discriminator_fields(child_model, child_dumped)
    elif isinstance(model, dict) and isinstance(dumped, dict):
        for key, child_model in model.items():
            if key in dumped:
                _inject_discriminator_fields(child_model, dumped[key])


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
            config_dict = stage_hyperparameters.model_dump(exclude_unset=True, exclude_computed_fields=True)
            _inject_discriminator_fields(stage_hyperparameters, config_dict)
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
        yaml_str = yaml.dump(stage_hyperparameters.model_dump(exclude_computed_fields=True)).rstrip("\n")
        _logger.debug(yaml_str)
