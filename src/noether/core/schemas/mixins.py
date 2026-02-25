#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from typing import Any

from pydantic import BaseModel, model_validator


class Shared:
    """Marker class to indicate a field should inherit shared values from the parent config."""


class InjectSharedFieldFromParentMixin(BaseModel):
    """Mixin to propagate shared fields from parent configuration to sub-configurations.

    Usage:
        class MyConfig(BaseModel, InjectSharedFieldFromParentMixin):
            sub_config: Annotated[SubConfigType, Shared]
    """

    @model_validator(mode="before")
    @classmethod
    def propagate_shared_fields(cls, data: Any) -> Any:
        """Propagates shared fields from parent config to sub-configurations if keys are identical."""
        if not isinstance(data, dict):
            return data

        # Iterate over all fields in the model
        for field_name, field_info in cls.model_fields.items():
            # Check if inheritance of shared fields is requested via Annotated[..., Shared]
            if not any(x is Shared for x in field_info.metadata):
                continue

            # Check if the field is a Pydantic model (i.e., a sub-config) that has model_fields attribute
            if isinstance(field_info.annotation, type) and issubclass(field_info.annotation, BaseModel):
                sub_model_type = field_info.annotation
            else:
                # Not a Pydantic model, skip
                continue

            # Get the sub-config data from the input dictionary
            sub_config_data = data.get(field_name)

            # Check if the sub-config is provided as a dictionary (i.e., not already a model instance)
            if isinstance(sub_config_data, dict):
                sub_model_fields = sub_model_type.model_fields.keys()

                # Iterate over all keys present in the parent data
                for parent_key, parent_value in data.items():
                    # If key exists in sub-config schema...
                    if parent_key in sub_model_fields:
                        # ...and is not the sub-config itself
                        if parent_key != field_name:
                            # ...and is NOT already defined in the specific sub-config data
                            if parent_key not in sub_config_data:
                                sub_config_data[parent_key] = parent_value

        return data
