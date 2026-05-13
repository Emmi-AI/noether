#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any, ClassVar

import torch

from noether.core.factory import Factory
from noether.core.schemas.lib import _RegistryBase
from noether.data.pipeline.collator import Collator, CollatorType

SampleProcessorType = Callable[[dict[str, Any]], dict[str, Any]]
BatchProcessorType = Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]


class PipelineConfig(_RegistryBase):
    _registry: ClassVar[dict[str, type]] = {}
    _type_field: ClassVar[str] = "kind"

    kind: str


class MultiStagePipeline(Collator):
    """A Collator that processes the list of samples into a batch in multiple stages:
        - sample_processors: Processing the data before collation on a per-sample level.
        - collators: Conversion from a list of samples into a batch (dict of usually tensors).
        - batch_processors: Processing after collation on a batch-level.
    Most of the work is usually done by the sample_processors. One or two collators, and batch processors are often not needed. However this depends on the use case.
    Example:

        .. testcode::

            from noether.data.pipeline.multistage import MultiStagePipeline

            pipeline = MultiStagePipeline()
            samples = [{"data": torch.tensor([1.0, 2.0])}, {"data": torch.tensor([3.0, 4.0])}]
            batch = pipeline(samples)
            print(batch["data"].shape)

        .. testoutput::

            torch.Size([2, 2])
    """

    def __init__(
        self,
        collators: dict[str, CollatorType] | list[CollatorType] | None = None,
        sample_processors: dict[str, SampleProcessorType] | list[SampleProcessorType] | None = None,
        batch_processors: dict[str, BatchProcessorType] | list[BatchProcessorType] | None = None,
    ):
        """
        Args:
            sample_processors: A list of callables that will be applied sequentially to pre-process on a per-sample level (e.g., subsample a pointcloud).
            collators: A list of callables that will be applied sequentially to convert the list of individual samples into a batched format. If None, the default PyTorch collator will be used.
            batch_processors: A list of callables that will be applied sequentially to process on a per-batch level.

        """

        self.sample_processors = Factory().create_list(sample_processors)
        if collators is not None and len(collators) > 0:
            self.collators = Factory().create_list(collators)
        else:
            from torch.utils.data import default_collate

            self.collators = [default_collate]
        self.batch_processors = Factory().create_list(batch_processors)

    def get_sample_processor(self, predicate: Callable[[Any], bool]) -> Any:
        """
        Retrieves a sample processor by a predicate function.
        Examples:
        - Search by type (assumes the sample processor type only occurs once in the list of sample processors)
          `pipeline.get_sample_processor(lambda p: isinstance(p, MySampleProcessorType))`
        - Search by type and member
          `pipeline.get_sample_processor(lambda p: isinstance(p, PointSamplingSampleProcessor) and "input_pos" in p.items)`

        Args:
            predicate: A function that is called for each processor and selects if this is the right one.

        Returns:
            Any: The matching sample processor.

        Raises:
            ValueError: If no matching sample processor are found, multiple matching sample processors are found or if there
                are no sample processors.
        """
        if len(self.sample_processors) == 0:
            raise ValueError("No sample processor matches predicate.")
        found_sample_processors = []
        for sample_processor in self.sample_processors:
            if predicate(sample_processor):
                found_sample_processors.append(sample_processor)
        if len(found_sample_processors) == 0:
            raise ValueError("No sample processor matches predicate.")
        if len(found_sample_processors) > 1:
            raise ValueError(f"Multiple sample processors match predicate ({found_sample_processors}).")
        return found_sample_processors[0]

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Applies a multi-stage collation pipeline to the loaded samples.

        Args:
            samples (list[dict[str, Any]]): List of individual samples retrieved from the dataset.

        Returns:
            Collated batch.
        """
        # pre-process on a sample level
        samples = [deepcopy(sample) for sample in samples]  # copy to avoid changing method input
        for sample_processor in self.sample_processors:
            for idx, sample in enumerate(samples):
                samples[idx] = sample_processor(sample)

        # create batch out of the samples
        batch = {}
        for batch_collator in self.collators:
            sub_batch = batch_collator(samples)
            # make sure that there is no overlap between collators
            for key, value in sub_batch.items():
                if key in batch:
                    raise ValueError(f"Key '{key}' already exists in batch. Collators must not overlap in keys.")
                batch[key] = value

        # process the batch
        for batch_processor in self.batch_processors:
            batch = batch_processor(batch)

        return batch

    def __str__(self) -> str:
        str_value = "MultiStagePipeline with"
        for component in ["sample_processors", "collators", "batch_processors"]:
            if (comp_len := len(getattr(self, component))) == 0:
                continue
            pluralised = component if comp_len > 1 else component[:-1]
            str_value += f" {comp_len} {pluralised}"
        return str_value

    def __repr__(self) -> str:
        return (
            "MultiStagePipeline(\n"
            f"  sample_processors={self.sample_processors},\n"
            f"  collators={self.collators},\n"
            f"  batch_processors={self.batch_processors}\n"
            ")"
        )
