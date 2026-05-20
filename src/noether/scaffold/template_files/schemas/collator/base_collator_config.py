#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.data.pipeline.multistage import PipelineConfig


class BasePipelineConfig(PipelineConfig):
    default_collate_modes: list[str] = ["x", "y"]
