#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from pydantic import Field

from noether.core.schemas.dataset import StandardDatasetConfig


class SimshiftHeatsinkConfig(StandardDatasetConfig):
    """Configuration for the SIMSHIFT Heatsink dataset.

    This dataset uses HDF5 files from the SIMSHIFT benchmark for unsupervised domain adaptation
    of neural surrogates for physical simulations.
    """

    kind: str | None = "noether.data.datasets.cfd.SimshiftHeatsinkDataset"

    difficulty: Literal["easy", "medium", "hard"] | None = Field(None)
    """Domain-gap difficulty level between source and target domains. If None, load all difficulties."""

    domain: Literal["source", "target"] | None = Field(None)
    """Which domain to load: source (in-distribution) or target (shifted). If None, load both."""

    splits_path: str | None = Field(default=None)
    """Path to the splits.json file. If None, defaults to {root}/splits.json."""

    metadata_path: str | None = Field(default=None)
    """Path to the metadata.csv file. If None, defaults to {root}/metadata.csv."""
