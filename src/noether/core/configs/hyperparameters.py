#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from noether.core.factory import class_constructor_from_class_path
from noether.core.schemas.lib import _RegistryBase
from noether.core.schemas.schema import ConfigSchema

_logger = logging.getLogger(__name__)


def _union_discriminator_name(field_info: Any) -> str | None:
    """Return the discriminator field name if ``field_info`` is a Pydantic
    discriminated-Union field (``Field(discriminator="kind")``), else ``None``.

    Pydantic stores the discriminator name on ``FieldInfo.discriminator`` and,
    when the discriminator was supplied via ``Annotated[..., Discriminator(...)]``,
    inside ``FieldInfo.metadata``. We check both.
    """
    name = getattr(field_info, "discriminator", None)
    if isinstance(name, str):
        return name
    for entry in getattr(field_info, "metadata", []) or []:
        if hasattr(entry, "discriminator") and isinstance(entry.discriminator, str):
            return entry.discriminator
    return None


def _inject_discriminator_fields(model: Any, dumped: Any) -> None:
    """Re-add discriminator fields (e.g. ``kind``) onto ``dumped`` for every
    discriminated config in ``model`` whose discriminator was stripped by
    ``model_dump(exclude_unset=True)``.

    Covers two discriminator styles:

    * :class:`_RegistryBase` subclasses with a class-level ``_type_field``
      (noether's ``Discriminated`` registry — e.g. ``CallBackBaseConfig.kind``).
    * Plain ``BaseModel`` variants of a Pydantic ``Union`` with
      ``Field(discriminator=...)`` — the discriminator value (typically a
      ``Literal`` with a default like ``FlowMatchingConfig.kind = "flow_matching"``)
      is stripped by ``exclude_unset=True`` and breaks union re-validation on
      reload. Detected by inspecting the parent field's ``discriminator``
      metadata so unrelated ``Literal`` enums (e.g. ``accelerator``) aren't
      forced into every dump.

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
        for field_name, field_info in type(model).model_fields.items():
            if field_name not in dumped:
                continue
            child_model = getattr(model, field_name)
            child_dumped = dumped[field_name]
            # Pydantic Union with discriminator: ensure the discriminator
            # value survives even when the variant's ``Literal`` field was
            # stripped by ``exclude_unset=True``.
            disc_name = _union_discriminator_name(field_info)
            if disc_name and isinstance(child_dumped, dict) and disc_name not in child_dumped:
                disc_value = getattr(child_model, disc_name, None)
                if disc_value is not None:
                    child_dumped[disc_name] = disc_value
            _inject_discriminator_fields(child_model, child_dumped)
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
            # ``serialize_as_any=True`` preserves subclass-specific fields when
            # the schema annotates a polymorphic field as the base class
            # (e.g. ``trainer.callbacks: list[CallBackBaseConfig]``) — without
            # it, Pydantic strips fields like ``OfflineLossCallbackConfig.dataset_key``
            # from the dump because they don't exist on the annotated base, and
            # the saved YAML fails to re-validate.
            config_dict = stage_hyperparameters.model_dump(
                exclude_unset=True, exclude_computed_fields=True, serialize_as_any=True
            )
            _inject_discriminator_fields(stage_hyperparameters, config_dict)
            config_dict["config_schema_kind"] = stage_hyperparameters.config_schema_kind
            yaml.dump(config_dict, f, sort_keys=False)

        _logger.info(f"Dumped resolved hyperparameters to {out_file_uri}")

    @staticmethod
    def load_resolved(file_uri: str | Path) -> ConfigSchema:
        """Inverse of :meth:`save_resolved` — load a previously-saved resolved config.

        Reads the YAML with ``yaml.full_load`` (preserves ``!!python/tuple``
        tags emitted for tuple-typed fields like dataset statistics), resolves
        the ``config_schema_kind`` marker written by ``save_resolved`` to the
        concrete :class:`ConfigSchema` subclass, and validates the rest of
        the dict through Pydantic.

        Args:
            file_uri: Path to the YAML file produced by ``save_resolved``
                (typically ``<run_dir>/hp_resolved.yaml``).

        Returns:
            The validated config schema instance.

        Raises:
            FileNotFoundError: if ``file_uri`` doesn't exist.
            pydantic.ValidationError: if the file is missing required fields
                or the config schema's validators reject it.
        """
        path = Path(file_uri)
        if not path.exists():
            raise FileNotFoundError(f"resolved hyperparameters file not found: {path}")
        with open(path) as f:
            config_dict = yaml.full_load(f)
        schema_kind = config_dict.pop("config_schema_kind", None)
        schema_cls: type[ConfigSchema] = class_constructor_from_class_path(schema_kind) if schema_kind else ConfigSchema  # type: ignore
        return schema_cls(**config_dict)

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
