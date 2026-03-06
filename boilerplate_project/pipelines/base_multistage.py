#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from boilerplate_project.schemas.collator.base_collator_config import BasePipelineConfig
from noether.data.pipeline import MultiStagePipeline
from noether.data.pipeline.collators import DefaultCollator


class BaseMultiStagePipeline(MultiStagePipeline):
    """A base multi-stage collator for testing purposes."""

    def __init__(self, pipeline_config: BasePipelineConfig):
        super().__init__(
            sample_processors=[],  # List of sample processors to be applied before collation
            collators=[DefaultCollator(items=list(set(pipeline_config.default_collate_modes)))],
            batch_processors=[],  # List of batch processors to be applied after collation
        )
