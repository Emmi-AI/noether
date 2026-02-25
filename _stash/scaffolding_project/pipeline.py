#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from pydantic import BaseModel

from noether.data.pipeline import MultiStagePipeline


class PipelineConfig(BaseModel):
    # define your pipeline config options here
    kind: str


class MultiStagePipeline(MultiStagePipeline):
    def __init__(self, pipeline_config: PipelineConfig, **kwargs):
        self.pipeline_config = pipeline_config
        # define your pipeline stages here
        super().__init__(sample_processors=[], collators=[], batch_processors=[], **kwargs)
