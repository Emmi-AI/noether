#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
from functools import partial
from typing import Any

import pydantic
from omegaconf import DictConfig, ListConfig

from noether.core.factory.utils import class_constructor_from_class_path


class Factory:
    """
    Base factory.

    Implements base structures for creating a single object, a list of objects and a dict
    of objects. The main difference between these methods are the default return values.
    As python does not like using an empty list/dict as default value for an argument,
    arguments are by default often None. By differentiating between these three types,
    one avoids none checks whenever the factory is called.

    * :meth:`create`: creates an object.
    * :meth:`create_list`: creates a list of objects.
    * :meth:`create_dict`: creates a dict of objects.

    For example, creating a list:
    .. code-block:: python

        class Example:
            def __init__(self, callbacks: list[Callback] | None = None):
                # automatic none check in create_list (this is how FactoryBase is implemented)
                self.callbacks = create_list(callbacks)
                # required none check after creating the list (this is how one could implement it without create_list)
                self.callbacks = create(callbacks) or []
    """

    def __init__(self, returns_partials: bool = False):
        self.logger = logging.getLogger(type(self).__name__)
        self.returns_partials = returns_partials

    def create(self, obj_or_kwargs: Any | dict[str, Any] | pydantic.BaseModel | None, **kwargs) -> Any | None:
        """Creates an object if the object is specified as dictionary. If the object was already instantiated, it will
        simply return the existing object. If `obj_or_kwargs` is None, None is returned.

        Args:
            obj_or_kwargs: Either an existing object (Any) or a description of how an object should be instantiated
                (dict[str, Any]).
            kwargs: Further kwargs that are passed when creating the object. These are often dependencies such as
                `UpdateCounter`, `PathProvider`, `MetricPropertyProvider`, ...

        Returns:
            The instantiated object.
        """

        if obj_or_kwargs is None or isinstance(obj_or_kwargs, dict | DictConfig) and len(obj_or_kwargs) == 0:
            return None

        if isinstance(obj_or_kwargs, pydantic.BaseModel):
            return self.instantiate(obj_or_kwargs, **kwargs)

        # instantiate object from dict
        if isinstance(obj_or_kwargs, dict | DictConfig):
            # Cast to dict to satisfy mypy
            dict_obj: dict[str, Any] = dict(obj_or_kwargs) if isinstance(obj_or_kwargs, DictConfig) else obj_or_kwargs
            obj_or_partial = self.instantiate(**dict_obj, **kwargs)
            if self.returns_partials:
                assert isinstance(obj_or_partial, partial | type)
            return obj_or_partial

        # e.g., optimizers return partials and don't instantiate the object
        if self.returns_partials:
            return obj_or_kwargs

        # check if obj_or_kwargs was already instantiated
        if not isinstance(obj_or_kwargs, partial | type):
            return obj_or_kwargs

        # instantiate
        return obj_or_kwargs(**kwargs)

    def create_list(
        self, collection: list[Any] | list[dict[str, Any]] | dict[str, Any] | list[pydantic.BaseModel] | None, **kwargs
    ) -> list[Any]:
        """Creates a list of object by calling the `create` function for every item in the collection. If `collection`
        is None, an empty list is returned.

        Args:
            collection: Either a list of configs how the objects should be instantiated.)
            kwargs: Further kwargs that are passed to all object instantiations. These are often dependencies such as
                `UpdateCounter`, `PathProvider`, `MetricPropertyProvider`, ...

        Returns:
            The instantiated list of objects or an empty list.
        """
        if collection is None:
            return []
        if isinstance(collection, dict):
            collection = list(collection.values())
        elif not isinstance(collection, list | ListConfig):
            raise NotImplementedError(f"invalid collection type {type(collection).__name__} (expected list or dict)")
        objs = [self.create(config, **kwargs) for config in collection]
        return objs

    def create_dict(
        self,
        collection: dict[str, Any] | dict[str, dict[str, Any]] | None,
        **kwargs,
    ) -> dict[str, Any]:
        """Creates a dict of object by calling the `create` function for every item in the collection. If `collection`
        is None, an empty dictionary is returned.

        Args:
            collection: Either a dict of existing objects (dict[str, Any]) or a dict of descriptions how the objects
                should be instantiated and what their identifier in the dict is (dict[str, dict[str, Any]]).
            kwargs: Further kwargs that are passed to all object instantiations. These are often dependencies such as
                `UpdateCounter`, `PathProvider`, `MetricPropertyProvider`, ...

        Returns:
            The instantiated dict of objects or an empty dict.
        """
        if collection is None:
            return {}
        if not isinstance(collection, dict):
            raise NotImplementedError(f"invalid collection type {type(collection).__name__} (expected dict)")
        objs = {key: self.create(constructor_kwargs, **kwargs) for key, constructor_kwargs in collection.items()}
        return objs

    def instantiate(self, object_config: Any = None, **kwargs) -> Any:
        """Instantiates an object based on its fully specified classpath.

        Args:
            object_config: Fully specified type of the object. For example: `"torch.optim.SGD"` or
                `"noether.core.callbacks.CheckpointCallback"`.
            kwargs: kwargs passed to the type when instantiating the object.

        Returns:
            The instantiated object.
        """
        if object_config is None and "kind" in kwargs:
            # some objects still need to be instantiated by using a dict, e.g. optimizers, this is done via the **kwargs, but this is a bit of a hack
            class_constructor = class_constructor_from_class_path(kwargs.pop("kind"))
            return class_constructor(**kwargs)
        else:
            class_constructor = class_constructor_from_class_path(object_config.kind)
            return class_constructor(object_config, **kwargs)
