#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import importlib
from abc import ABC
from functools import partial
from typing import Any, ClassVar, TypeVar, get_type_hints

from pydantic import BaseModel, BeforeValidator


class _RegistryBase(BaseModel, ABC):
    """
    Internal base class for all registry-based configs.

    Provides auto-registration via __init_subclass__.
    Not meant to be used directly - use specific config base classes instead.
    """

    _registry: ClassVar[dict[str, type[BaseModel]]]
    _type_field: ClassVar[str] = "type"


T = TypeVar("T", bound=_RegistryBase)


def resolve_config_class[T: _RegistryBase](kind: str, base_cls: type[T]) -> type[T]:
    """Resolve a config class from a dotted kind string.

    Resolution order:
        1. If the imported class is already a ``base_cls`` subclass, return it directly.
        2. Check for a ``_config_class`` attribute (set by ``@ConfiguredBy``).
        3. Inspect ``__init__`` type hints for the first parameter that is a ``base_cls`` subclass.

    Args:
        kind: fully qualified dotted path (e.g., ``"noether.training.trainers.WeightedLossTrainer"``).
        base_cls: the base config class to resolve against.

    Raises:
        ValueError: if the config class cannot be determined.
    """
    module_name, class_name = kind.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    # Already a config subclass:
    if isinstance(cls, type) and issubclass(cls, base_cls):
        return cls

    # Explicit _config_class attribute (@ConfiguredBy):
    if hasattr(cls, "_config_class") and issubclass(cls._config_class, base_cls):
        return cls._config_class  # type: ignore[no-any-return]

    # First __init__ type hint that is a base_cls subclass:
    try:
        hints = get_type_hints(cls.__init__)
    except Exception:
        hints = {}
    for hint in hints.values():
        if isinstance(hint, type) and issubclass(hint, base_cls):
            return hint

    raise ValueError(
        f"Cannot resolve config class for '{kind}' against {base_cls.__name__}. "
        "Use the @ConfiguredBy decorator to specify the configuration class."
    )


def Discriminated(registry_cls: type[_RegistryBase]):
    """
    Returns a BeforeValidator that instantiates components based on their registry keys.
    Usage: field: Annotated[Any, Discriminated(MyComponent)]
    """

    return BeforeValidator(partial(_discriminated_validator, registry_cls=registry_cls))


def _discriminated_validator(item, registry_cls: type[_RegistryBase]) -> Any:
    # Skip if already instantiated or not a dict
    if not isinstance(item, dict):
        return item

    # If type field is present, try to find class
    if registry_cls._type_field not in item:
        raise ValueError(
            f"Missing required field '{registry_cls._type_field}' for discriminated union of {registry_cls.__name__}. Found keys: {list(item.keys())}"
        )

    type_key = item[registry_cls._type_field]

    # 1. Lookup in registry
    if type_key in registry_cls._registry:
        return registry_cls._registry[type_key].model_validate(item)

    # 2. Try dynamic import (for external components)
    if "." in type_key:
        config_class = resolve_config_class(type_key, registry_cls)
        return config_class.model_validate(item)

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
