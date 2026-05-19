#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from pydantic import Field

from noether.data.pipeline import MultiStagePipeline
from noether.data.pipeline.collators import (
    DefaultCollator,
)
from noether.data.pipeline.multistage import PipelineConfig
from noether.data.pipeline.sample_processors import (
    ConcatTensorSampleProcessor,
    PointSamplingSampleProcessor,
)


class DevelopmentPipelineConfig(PipelineConfig):
    num_points: int = Field(default=56, description="Number of points to sample from the point cloud.")


class DevelopmentPipeline(MultiStagePipeline):
    def __init__(self, config: DevelopmentPipelineConfig):
        self.config = config

        super().__init__(
            sample_processors=[
                PointSamplingSampleProcessor(items=["x", "y", "z"], num_points=self.config.num_points),
                ConcatTensorSampleProcessor(items=["x", "z"], target_key="x_z", dim=1),
            ],
            batch_processors=[],
            collators=[DefaultCollator(items=["x_z", "y"])],
        )
