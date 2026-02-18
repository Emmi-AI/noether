#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schemas.dataset import CAEMLDatasetConfig
from tutorial.schemas.pipelines.aero_pipeline_config import AeroCFDPipelineConfig


class AeroDatasetConfig(CAEMLDatasetConfig):
    pipeline: AeroCFDPipelineConfig
    filter_categories: tuple[str] | None = None
