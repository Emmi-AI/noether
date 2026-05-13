#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from pydantic import Field

from noether.data.base.dataset import DatasetBaseConfig


class SimshiftHeatsinkConfig(DatasetBaseConfig):
    """Configuration for the SIMSHIFT Heatsink dataset.

    This dataset uses HDF5 files from the SIMSHIFT benchmark for unsupervised domain adaptation
    of neural surrogates for physical simulations.

    If ``root`` is not set, the dataset is automatically downloaded from the HuggingFace Hub
    (``simshift/SIMSHIFT_data``, ``heatsink.zip``) and cached locally.
    """

    kind: str | None = "noether.data.datasets.cfd.SimshiftHeatsinkDataset"

    root: str | None = None  # type: ignore[assignment]
    """Root directory of the dataset. If None, auto-downloads from HuggingFace Hub."""

    split: Literal["train", "val", "test"]
    """Which split of the dataset to use. Must be one of "train", "val", or "test"."""

    difficulty: Literal["easy", "medium", "hard"] | None = Field(None)
    """Domain-gap difficulty level between source and target domains. If None, load all difficulties."""

    domain: Literal["source", "target"] | None = Field(None)
    """Which domain to load: source (in-distribution) or target (shifted). If None, load both."""

    splits_path: str | None = Field(default=None)
    """Path to the splits.json file. If None, defaults to {root}/splits.json."""

    metadata_path: str | None = Field(default=None)
    """Path to the metadata.csv file. If None, defaults to {root}/metadata.csv."""
