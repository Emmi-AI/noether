#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import functools
import logging
from collections.abc import Iterator
from typing import Any

from torch.utils.data import Dataset as TorchDataset

from noether.core.factory import Factory
from noether.core.schemas.dataset import DatasetBaseConfig
from noether.data.pipeline import Collator, MultiStagePipeline
from noether.data.preprocessors import ComposePreProcess, PreProcessor


def with_normalizers(_func_or_key: str | Any | None = None):
    """Decorator to apply a normalizer to the output of a getitem_* function of the implemented Dataset class.

    This decorator will look for a normalizer registered under the specified key and apply it to the output
    of the decorated function. If no key is provided, the key is automatically inferred from the function name
    by removing the 'getitem_' prefix.

    Example usage:

    .. code-block:: python

        # Inferred key: "surface_pressure"
        @with_normalizers
        def getitem_surface_pressure(self, idx):
            return torch.load(f"{self.path}/surface_pressure/{idx}.pt")


        # Explicit key: "pressure"
        @with_normalizers("pressure")
        def getitem_surface_pressure(self, idx):
            return torch.load(f"{self.path}/surface_pressure/{idx}.pt")

    Args:
        _func_or_key: The normalizer key (str) or the function being decorated.
            If used as `@with_normalizers` (no arguments), this will be the decorated function.
            If used as `@with_normalizers("key")`, this will be the string key.

    Returns:
        The decorated function with normalization applied.

    Raises:
        ValueError: If the normalizer key cannot be resolved from the function name.
        AttributeError: If the class instance does not have a 'normalizers' attribute.
        KeyError: If the requested normalizer key is not found in the 'normalizers' dictionary.
    """

    def resolve_normalizer_key(fn: Any) -> str:
        # Allow usage as @with_normalizers or @with_normalizers("key")
        if isinstance(_func_or_key, str):
            return _func_or_key
        fn_name = str(fn.__name__)
        if fn_name.startswith("getitem_"):
            return fn_name[len("getitem_") :]
        raise ValueError(
            "Could not resolve normalizer_key: either provide it explicitly or ensure the function name follows 'getitem_{key}' pattern."
        )

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            normalizer_key = resolve_normalizer_key(fn)
            data = fn(self, *args, **kwargs)
            if self.compute_statistics:
                return data
            try:
                registry_attribute = "normalizers"
                normalizers = getattr(self, registry_attribute)
            except AttributeError as exc:
                raise AttributeError(
                    f"{self.__class__.__name__}.{registry_attribute} not found; "
                    f"required for with_normalizers('{normalizer_key}') method to have the normalizers attribute"
                ) from exc
            try:
                normalizer = normalizers[normalizer_key]
            except KeyError:
                available_keys = list(normalizers.keys())
                raise KeyError(
                    f"Normalizer key '{normalizer_key}' not found. Normalizers are available for the following getitem_ methods: {available_keys}"
                ) from None
            data = normalizer(data)
            return data

        return wrapper

    if callable(_func_or_key):
        return decorator(_func_or_key)

    return decorator


class Dataset(TorchDataset):
    """Noether dataset implementation, which is a wrapper around torch.utils.data.Dataset that can hold a dataset_config_provider.
    A dataset should map a key (i.e., an index) to its corresponding data.
    Each sub-class should implement individual getitem_* methods, where * is the name of an item in the dataset.
    Each getitem_* method loads an individual tensor/data sample from disk.
    For example, if you dataset consists of images and targets/labels (stored as tensors), a getitem_image(idx) and getitem_target(idx) method should be implemented in the dataset subclass.
    The __getitem__ method of this class will loop over all the individual getitem_* methods implemented by the child class and return their results.
    Optionally it is possible to configure which getitem methods are called.

    Example: Image classification datasets

    .. code-block:: python

        class CarAeroDynamicsDataset(Dataset):
            def __init__(self, dataset_config, dataset_normalizers, **kwargs):
                super().__init__(dataset_config=dataset_config, **kwargs)
                self.path = dataset_config.path

            def __len__(self):
                return 100  # Example length

            def getitem_surface_pressure(self, idx):
                # Load surface pressure tensor
                return torch.load(f"{self.path}/surface_pressure_tensor/{idx}.pt")

            def getitem_surface_geometry(self, idx):
                # Load surface geometry tensor
                return torch.load(f"{self.path}/surface_geometry_tensor/{idx}.pt")


        dataset = CarAeroDynamicsDataset("path/to/dataset")
        sample0 = dataset[0]
        surface_pressure_0 = sample0["surface_pressure"]
        surface_geometry_0 = sample0["surface_geometry"]

    Data from a getitem method should be normalized in many cases. To apply normalization, add a the decorator function to the getitem method.
    For example:

    .. code-block:: python

        @with_normalizers("surface_pressure")
        def getitem_surface_pressure(self, idx):
            # Load surface pressure tensor
            return torch.load(f"{self.path}/surface_pressure_tensor/{idx}.pt")

    "surface_pressure" is the key in the self.normalizers dictionary, this key maps to a preprocessor that should implement the correct data normalization.

    Example configuration for dataset normalizers:

    .. code-block:: yaml

        # dummy example configuration for an image classification
        dataset:
            kind: noether.data.datasets.CarAeroDynamicsDataset
            pipeline:  # configure the data pipeline to collate individual samples into batches
            dataset_normalizers:
                surface_pressure:
                    - kind: noether.data.preprocessors.normalizers.MeanStdNormalization
                      mean: [1., 2., 3.]
                      std: [0.1, 0.2, 0.3]
    """

    def __init__(
        self,
        dataset_config: DatasetBaseConfig,
    ):
        """

        Args:
            dataset_config: Configuration for the dataset. See :class:`~noether.core.schemas.dataset.DatasetBaseConfig`
                for available options including dataset normalizers.
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)
        self._pipeline: Collator | MultiStagePipeline | None = None
        self.config = dataset_config
        self.normalizers: dict[str, ComposePreProcess] = {}
        self.compute_statistics = False
        if dataset_config.dataset_normalizers:
            for key, normalizer_configs in dataset_config.dataset_normalizers.items():
                if not isinstance(normalizer_configs, list):
                    normalizer_configs = [normalizer_configs]
                preprocessors: list[PreProcessor] = []
                for normalizer_config in normalizer_configs:
                    preprocessors.append(Factory().instantiate(normalizer_config, normalization_key=key))
                self.normalizers[key] = ComposePreProcess(normalization_key=key, preprocessors=preprocessors)

    @property
    def pipeline(self) -> Collator | None:
        """Returns the pipeline for the dataset."""
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline: Collator) -> None:
        """Sets the pipeline for the dataset."""
        if not isinstance(pipeline, Collator):
            raise TypeError(f"Expected Collator instance, got {type(pipeline)}")
        self._pipeline = pipeline

    def __len__(self) -> int:
        raise NotImplementedError("__len__ method must be implemented")

    def __getitem__(self, idx: int) -> Any:
        """Calls all implemented getitem methods and returns the results

        Returns:
            dict[key, Any]: dictionary of all getitem result
        """
        result = dict(index=idx)
        getitem_names = self.get_all_getitem_names()
        for getitem_name in getitem_names:
            getitem_fn = getattr(self, getitem_name)
            result[getitem_name[len("getitem_") :]] = getitem_fn(idx)
        return result

    def __iter__(self) -> Iterator[Any]:
        """torch.utils.data.Dataset doesn't define __iter__ which makes 'for sample in dataset' run endlessly.

        Returns:
            Iterator[Any]: an iterator of the type that would be returned by __getitem__
        """
        for i in range(len(self)):
            yield self[i]

    def get_all_getitem_names(self) -> list[str]:
        """Returns all names of getitem functions that are implemented. E.g., image classification has getitem_x and
        getitem_class -> the result will be ["x", "class"]."""
        return [attr for attr in dir(self) if attr.startswith("getitem_") and callable(getattr(self, attr))]
