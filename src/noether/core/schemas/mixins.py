#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from typing import Any, ClassVar

from pydantic import BaseModel, model_validator


class SharedFieldPropagationMixin:
    """Mixin to propagate shared fields from parent configuration to sub-configurations.

    Usage:
        class MyConfig(BaseModel, SharedFieldPropagationMixin):
            _SHARED_CONFIGS_MAP = {"sub_config": SubConfigType}
    """

    _SHARED_CONFIGS_MAP: ClassVar[dict[str, type[BaseModel]]] = {}
    """Map of field names to their expected Pydantic model classes."""

    @model_validator(mode="before")
    @classmethod
    def propagate_shared_fields(cls, data: Any) -> Any:
        """Propagates shared fields from parent config to sub-configurations if keys are identical."""
        if not isinstance(data, dict):
            return data

        # If no shared configs map is defined, return data as is
        # We access the attribute safely to allow usage where it might not be defined
        shared_map = getattr(cls, "_SHARED_CONFIGS_MAP", {})
        if not shared_map:
            return data

        # Iterate over each sub-configuration defined in the map
        for config_key, config_cls in shared_map.items():
            sub_config_data = data.get(config_key)

            # We only act if the sub-config is present and is a dictionary (raw data)
            if isinstance(sub_config_data, dict):
                # Get all fields defined in the sub-config schema
                sub_model_fields = config_cls.model_fields.keys()

                # Iterate over all keys present in the parent data
                for parent_key, parent_value in data.items():
                    # If the key exists in the sub-config schema...
                    if parent_key in sub_model_fields:
                        # ...and acts as a source of truth (that is, not one of the sub-configs itself)
                        if parent_key not in shared_map:
                            # ...and is NOT already defined in the specific sub-config data
                            if parent_key not in sub_config_data:
                                sub_config_data[parent_key] = parent_value

        return data
