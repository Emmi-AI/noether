#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schemas.dataset import StandardDatasetConfig
from tutorial.schemas.pipelines.aero_pipeline_config import AeroCFDPipelineConfig


class AeroDatasetConfig(StandardDatasetConfig):
    pipeline: AeroCFDPipelineConfig
    filter_categories: tuple[str] | None = None
