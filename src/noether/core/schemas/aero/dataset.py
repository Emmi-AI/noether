#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from noether.core.schemas.aero.pipeline import AeroCFDPipelineConfig
from noether.core.schemas.dataset import StandardDatasetConfig


class AeroDatasetConfig(StandardDatasetConfig):
    """Dataset configuration for aerodynamic CFD datasets."""

    pipeline: AeroCFDPipelineConfig
    filter_categories: tuple[str, ...] | None = None
