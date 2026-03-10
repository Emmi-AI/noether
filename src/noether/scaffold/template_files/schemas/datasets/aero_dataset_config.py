#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __PROJECT__.schemas.pipelines.aero_pipeline_config import AeroCFDPipelineConfig

from noether.core.schemas.dataset import StandardDatasetConfig


class AeroDatasetConfig(StandardDatasetConfig):
    pipeline: AeroCFDPipelineConfig
    filter_categories: tuple[str] | None = None
