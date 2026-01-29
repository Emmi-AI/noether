#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Any

from noether.core.factory.base import Factory
from noether.core.schemas.dataset import DatasetBaseConfig


class DatasetFactory(Factory):
    """
    Specialized factory for datasets. Next to the standard instantiation of datasets, it also supports wrapping of the dataset inside dataset wrappers.

    Example config:

    .. code-block:: yaml

        kind: noether.core.factory.DatasetFactory
        dataset_wrappers:
          - kind: noether.data.base.wrappers.SomeDatasetWrapper
            param1: value1
          - kind: noether.data.base.wrappers.AnotherDatasetWrapper
            param2: value2
    """

    def __init__(
        self,
        dataset_wrapper_factory: Factory | None = None,
    ):
        super().__init__()
        self.dataset_wrapper_factory = dataset_wrapper_factory or Factory()

    def instantiate(self, dataset_config: DatasetBaseConfig, **kwargs) -> Any:  # type: ignore[override]
        """Instantiates the dataset based on the provided configuration.

        Args:
            dataset_config: Configuration for the dataset. See
                :class:`~noether.core.schemas.dataset.DatasetBaseConfig`
                for available options including dataset wrappers.
            **kwargs: Additional keyword arguments provided to the parent class.

        Returns:
            The instantiated dataset, possibly wrapped in dataset wrappers specified in the configuration.
        """

        dataset_wrappers = dataset_config.dataset_wrappers
        dataset = super().instantiate(dataset_config)

        if dataset_config.included_properties is not None or dataset_config.excluded_properties is not None:
            from noether.data.base.wrappers.property_subset import PropertySubsetWrapper  # avoid circular import

            dataset = PropertySubsetWrapper.from_included_excluded(
                dataset,
                included_properties=dataset.config.included_properties,
                excluded_properties=dataset.config.excluded_properties,
            )
        # wrap with dataset_wrappers
        if dataset_wrappers is not None:
            if not isinstance(dataset_wrappers, list):
                raise ValueError("dataset_wrappers must be a list of dataset wrapper configs")

            for dataset_wrapper_config in dataset_wrappers:
                dataset_wrapper_kind = dataset_wrapper_config.kind
                self.logger.info(f" Instantiating dataset_wrapper: {dataset_wrapper_kind}")
                dataset = self.dataset_wrapper_factory.instantiate(dataset_wrapper_config, dataset=dataset)
        return dataset
