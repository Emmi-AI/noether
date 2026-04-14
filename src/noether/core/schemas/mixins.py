#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import typing
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, model_validator
from pydantic.fields import FieldInfo


class Shared:
    """Marker class to indicate a field should inherit shared values from the parent config."""


def _has_marker(field_info: FieldInfo) -> bool:
    """Extract a BaseModel subclass from an annotation, handling Optional/Union types."""
    metadata = field_info.metadata
    # Handle Union types like `X | None` or `Optional[X]`
    if get_origin(field_info.annotation) is Union:
        for arg in get_args(field_info.annotation):
            metadata.extend(getattr(arg, "__metadata__", []))
    return any(x is Shared for x in metadata)


def _extract_base_model(annotation: Any) -> type[BaseModel] | None:
    """Extract a BaseModel subclass from an annotation, handling Optional/Union types."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    # Handle Union types like `X | None` or `Optional[X]`
    if get_origin(annotation) is Union:
        for arg in get_args(annotation):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
            if hasattr(arg, "__origin__") and issubclass(arg.__origin__, BaseModel):
                return typing.cast("type[BaseModel]", arg.__origin__)
    return None


class InjectSharedFieldFromParentMixin(BaseModel):
    """Mixin to propagate shared fields from parent configuration to sub-configurations.

    Supports recursive/nested injection across multiple levels of configuration hierarchy.

    Usage:
        class MyConfig(BaseModel, InjectSharedFieldFromParentMixin):
            sub_config: Annotated[SubConfigType, Shared]
    """

    @model_validator(mode="before")
    @classmethod
    def propagate_shared_fields(cls, data: Any) -> Any:
        """Propagates shared fields from parent config to sub-configurations recursively."""
        if not isinstance(data, dict):
            return data

        cls._inject_shared_fields_recursive(data, cls)
        return data

    @classmethod
    def _inject_shared_fields_recursive(cls, parent_data: dict, parent_model_type: type[BaseModel]) -> None:
        """Recursively inject shared fields into nested sub-configurations.

        Args:
            parent_data: Dictionary containing parent configuration data
            parent_model_type: Pydantic model type of the parent configuration
        """
        # Iterate over all fields in the parent model
        for field_name, field_info in parent_model_type.model_fields.items():
            # Check if inheritance of shared fields is requested via Annotated[..., Shared]
            if not _has_marker(field_info):
                continue

            # Check if the field is a Pydantic model (i.e., a sub-config)
            sub_model_type = _extract_base_model(field_info.annotation)
            if sub_model_type is None:
                continue

            # Get the sub-config data from the parent dictionary
            sub_config_data = parent_data.get(field_name)

            # Check if the sub-config is provided as a dictionary
            if isinstance(sub_config_data, dict):
                sub_model_fields = sub_model_type.model_fields.keys()

                # Inject fields from parent to this sub-config
                for parent_key, parent_value in parent_data.items():
                    # If key exists in sub-config schema...
                    if parent_key in sub_model_fields:
                        # ...and is NOT already defined in the specific sub-config data
                        if parent_key not in sub_config_data:
                            sub_config_data[parent_key] = parent_value

                # Recursively inject into nested sub-configs
                # Pass both the sub_config_data and parent_data to allow grandparent->grandchild injection
                cls._inject_nested_shared_fields(sub_config_data, sub_model_type, parent_data)

    @classmethod
    def _inject_nested_shared_fields(
        cls, config_data: dict, config_model_type: type[BaseModel], ancestor_data: dict
    ) -> None:
        """Inject shared fields into nested sub-configs, including from ancestor configs.

        Args:
            config_data: Dictionary containing current configuration data
            config_model_type: Pydantic model type of the current configuration
            ancestor_data: Dictionary containing ancestor configuration data for fallback injection
        """
        # Iterate over all fields in the current model
        for field_name, field_info in config_model_type.model_fields.items():
            # Check if inheritance of shared fields is requested via Annotated[..., Shared]
            if not _has_marker(field_info):
                continue

            # Check if the field is a Pydantic model (i.e., a sub-config)
            sub_model_type = _extract_base_model(field_info.annotation)
            if sub_model_type is None:
                continue

            # Get the nested sub-config data
            sub_config_data = config_data.get(field_name)

            if isinstance(sub_config_data, dict):
                sub_model_fields = sub_model_type.model_fields.keys()

                # First try to inject from immediate parent (config_data)
                for config_key, config_value in config_data.items():
                    if config_key in sub_model_fields and config_key != field_name:
                        if config_key not in sub_config_data:
                            sub_config_data[config_key] = config_value

                # Then try to inject from ancestor if still missing
                for ancestor_key, ancestor_value in ancestor_data.items():
                    if ancestor_key in sub_model_fields and ancestor_key != field_name:
                        if ancestor_key not in sub_config_data:
                            sub_config_data[ancestor_key] = ancestor_value

                # Continue recursively
                cls._inject_nested_shared_fields(sub_config_data, sub_model_type, ancestor_data)
