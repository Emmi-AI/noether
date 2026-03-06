#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import importlib
from abc import ABC
from functools import partial
from typing import Any, ClassVar, Literal, get_type_hints

from pydantic import BaseModel, BeforeValidator


class _RegistryBase(BaseModel, ABC):
    """
    Internal base class for all registry-based configs.

    Provides auto-registration via __init_subclass__.
    Not meant to be used directly - use specific config base classes instead.
    """

    _registry: ClassVar[dict[str, type[BaseModel]]]
    _type_field: ClassVar[str] = "type"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Check if this class is a registry root (defines its own _registry)
        if "_registry" in cls.__dict__:
            return

        if issubclass(cls, ABC):
            return  # Don't register abstract base classes

        # Find the registry root
        root = None
        for base in cls.__mro__:
            if base is cls:
                continue
            if issubclass(base, _RegistryBase) and "_registry" in base.__dict__:
                root = base
                break

        if root:
            # Register this subclass
            key = getattr(cls, root._type_field, None)
            if key is None:
                raise ValueError(
                    f"Class {cls.__name__} must define a class variable '{root._type_field}' for registration."
                )

            if not hasattr(cls, "__annotations__"):
                cls.__annotations__ = {}
            cls.__annotations__[root._type_field] = Literal[key]
            setattr(cls, root._type_field, key)

            root._registry[key] = cls


def Discriminated(registry_cls: type[_RegistryBase]):
    """
    Returns a BeforeValidator that instantiates components based on their registry keys.
    Usage: field: Annotated[Any, Discriminated(MyComponent)]
    """

    return BeforeValidator(partial(_discrimnated_validator, registry_cls=registry_cls))


def _discrimnated_validator(item, registry_cls: type[_RegistryBase]) -> Any:
    # Skip if already instantiated or not a dict
    if not isinstance(item, dict):
        return item

    # If type field is present, try to find class
    if registry_cls._type_field in item:
        type_key = item[registry_cls._type_field]

        # 1. Lookup in registry
        if type_key in registry_cls._registry:
            return registry_cls._registry[type_key].model_validate(item)

        # 2. Try dynamic import (for external components)
        if "." in type_key:
            module_name, class_name = type_key.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls_ = getattr(module, class_name)
            if issubclass(cls_, registry_cls):
                return cls_.model_validate(item)

            config_class = None
            # get class from first __init__ argument
            if hasattr(cls_, "__init__"):
                try:
                    init_params = get_type_hints(cls_.__init__)
                except NameError as e:
                    raise ImportError(
                        f"Failed to get type hints for {cls_}: {e}. Ensure all dependencies are installed and imports are correct."
                    ) from e
                if init_params:
                    config_class = next(iter(init_params.values()))

            if hasattr(cls_, "_config_class"):
                config_class = (
                    cls_._config_class
                    if config_class is None or issubclass(cls_._config_class, config_class)
                    else config_class
                )

            if config_class and issubclass(config_class, registry_cls):
                return config_class.model_validate(item)
            else:
                raise ValueError(
                    f"Unknown type key '{type_key}' for {registry_cls}. Use the @ConfiguredBy({class_name}) decorator to specify the configuration class."
                )

    return item


def ConfiguredBy(config_class: type[BaseModel]):
    """
    Decorator to mark a class as being configured by a specific config class.
    Usage:
        @ConfiguredBy(MyConfig)
        class MyClass:
            ...
    """

    def decorator(cls):
        cls._config_class = config_class
        return cls

    return decorator
