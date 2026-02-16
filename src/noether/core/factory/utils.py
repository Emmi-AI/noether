#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import importlib
import inspect
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def all_ctor_kwarg_names(cls: type) -> set[str]:
    """Returns all names of kwargs in the constructor of a type (including the kwargs of all parent classes).

    Args:
        cls: Type from which to retrieve the names of constructor kwargs.

    Returns:
        A set of names of constructor kwargs of the type.
    """
    return _all_ctor_kwarg_names(cls=cls, result=set())


def _all_ctor_kwarg_names(cls: type, result: set[str]) -> set[str]:
    for name in inspect.signature(cls).parameters.keys():
        result.add(name)
    if cls.__base__ is not None:
        _all_ctor_kwarg_names(cls.__base__, result)
    return result


def class_constructor_from_class_path(class_path: str) -> Callable[..., Any]:
    """Creates a callable that constructs an object from a classpath. This callable is either a ``type`` (if no further
    kwargs are needed to be passed) or a :func:`~functools.partial` otherwise. This is equivalent to Hydra instantiation with ``_target_``, which is also based on class paths.

    Args:
        class_path: Fully specified module path of the object. For example: ``"torch.optim.SGD"`` or
            ``"noether.core.callbacks.checkpoint.checkpoint.CheckpointCallback"``.

    Returns:
        A callable that constructs the object.
    """
    # split into module and class name
    # e.g. torch.nn.Linear -> module_name=torch.nn type_name=Linear

    split = class_path.split(".")
    assert len(split) > 1, f"invalid path to class ({class_path}) use MODULE_NAME.CLASS_NAME"
    module_name = ".".join(split[:-1])
    type_name = split[-1]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        logger.error(
            f"Could not find module {module_name}. "
            "Check if you spelled everything right (case sensitive). "
            "You might have forgotten to install the package that provides this module."
        )
        raise e

    try:
        ctor: Callable[..., Any] = getattr(module, type_name)
    except AttributeError as e:
        logger.error(
            f"Could not find type {type_name} in module {module}. "
            "Check if you spelled everything right (case sensitive). "
            "You might have forgotten to import the class in the __init__.py file."
        )
        raise e

    if not callable(ctor):
        raise TypeError(f"Retrieved object {ctor} from {class_path} is not callable.")
    return ctor
