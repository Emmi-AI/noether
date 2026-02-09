#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
from functools import partial
from typing import Any

import pydantic
from omegaconf import DictConfig, ListConfig

from noether.core.factory.utils import class_constructor_from_class_path


class Factory:
    """Base factory. Implements base structures for creating a single object, a list of objects and a dict
    of objects.

    For example, creating a list:

    .. code-block:: python

        class Example:
            def __init__(self, callbacks: list[CallbackConfig] | None = None):
                # automatic none check in create_list (this is how Factory is implemented)
                self.callbacks = Factory().create_list(callbacks)
                # required none check after creating the list (this is how one could implement it without create_list)


    Objects can be instantiated from either:

    * :class:`~pydantic.BaseModel`: In this case, the ``kind`` field is used to determine the class path of the
      object to be instantiated. The full pydantic model is passed to the constructor.

      .. code-block:: python

          class ExampleConfig(pydantic.BaseModel):
              kind: str
              param1: int
              param2: str


          class ExampleObject:
              def __init__(self, config: ExampleConfig):
                  self.param1 = config.param1
                  self.param2 = config.param2
                  # kind is also in the config, but usually not needed in the object itself


          example_config = ExampleConfig(kind="path.to.ExampleObject", param1=42, param2="hello")
          example_object = Factory().create(example_config)

      Using :class:`~pydantic.BaseModel` is the preferred way of instantiating objects as it provides type safety
      and validation.

    * Dictionary: In this case, the ``kind`` key is used to determine the class path of the object to be
      instantiated. The full dictionary is passed to the constructor. However, almost all classes in Noether take a
      config object as input to the constructor. This approach will only work for custom classes that take named
      arguments in the constructor.

      .. code-block:: python

          example_dict = {"kind": "path.to.ExampleObject", "param1": 42, "param2": "hello"}


          class ExampleObject:
              # constructor takes named arguments directly, and kind popped before passing to constructor
              def __init__(self, param1: int, param2: str):
                  self.param1 = param1
                  self.param2 = param2


          example_object = Factory().create(example_dict)
    """

    def __init__(self, returns_partials: bool = False):
        self.logger = logging.getLogger(type(self).__name__)
        self.returns_partials = returns_partials

    def create(self, obj_or_kwargs: Any | dict[str, Any] | pydantic.BaseModel | None, **kwargs) -> Any | None:
        """Creates an object if the object is specified as dictionary. If the object was already instantiated, it will
        simply return the existing object. If ``obj_or_kwargs`` is ``None``, ``None`` is returned.

        Args:
            obj_or_kwargs: Either an existing object or a description of how an object should be instantiated
                (dict or :class:`~pydantic.BaseModel`).
            kwargs: Further kwargs that are passed when creating the object. These are often dependencies such as
                :class:`~noether.core.utils.training.counter.UpdateCounter`,
                :class:`~noether.core.providers.path.PathProvider`,
                :class:`~noether.core.providers.metric_property.MetricPropertyProvider`, etc.

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
        """Creates a list of objects by calling the :meth:`create` function for every item in the collection.

        If ``collection`` is ``None``, an empty list is returned.

        Args:
            collection: Either a list of configs how the objects should be instantiated.
            kwargs: Further kwargs that are passed to all object instantiations. These are often dependencies such as
                :class:`~noether.core.utils.training.counter.UpdateCounter`,
                :class:`~noether.core.providers.path.PathProvider`,
                :class:`~noether.core.providers.metric_property.MetricPropertyProvider`, etc.

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
        """Creates a dict of objects by calling the :meth:`create` function for every item in the collection.

        If ``collection`` is ``None``, an empty dictionary is returned.

        Args:
            collection: Either a dict of existing objects or a dict of descriptions how the objects
                should be instantiated and what their identifier in the dict is.
            kwargs: Further kwargs that are passed to all object instantiations. These are often dependencies such as
                :class:`~noether.core.utils.training.counter.UpdateCounter`,
                :class:`~noether.core.providers.path.PathProvider`,
                :class:`~noether.core.providers.metric_property.MetricPropertyProvider`, etc.

        Returns:
            The instantiated dict of objects or an empty dict.
        """
        if collection is None:
            return {}
        if not isinstance(collection, dict):
            raise NotImplementedError(f"invalid collection type {type(collection).__name__} (expected dict)")
        objs = {key: self.create(constructor_kwargs, **kwargs) for key, constructor_kwargs in collection.items()}
        return objs

    def instantiate(self, object_config: pydantic.BaseModel | None = None, **kwargs) -> Any:
        """Instantiates an object based on its fully specified classpath.

        Args:
            object_config: Configuration containing the fully specified type of the object in the ``kind`` field.
                For example: ``"torch.optim.SGD"`` or ``"noether.core.callbacks.CheckpointCallback"``.
            kwargs: kwargs passed to the type when instantiating the object.

        Returns:
            The instantiated object.
        """

        if object_config is None and "kind" in kwargs:
            # some objects still need to be instantiated by using a dict, e.g. optimizers, this is done via the **kwargs, but this is a bit of a hack
            class_constructor = class_constructor_from_class_path(kwargs.pop("kind"))
            return class_constructor(**kwargs)
        else:
            class_constructor = class_constructor_from_class_path(object_config.kind)  # type: ignore [union-attr]
            return class_constructor(object_config, **kwargs)
