#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from noether.data.base.dataset import DatasetSplitIDs, StandardDatasetConfig
from noether.data.datasets.cfd.emmi_wing.dataset import EmmiWingDataset
from noether.data.datasets.cfd.emmi_wing.split_hf import WingHFSplitIDs

logger = logging.getLogger(__name__)

HF_REPO_ID = "EmmiAI/Emmi-Wing"


class EmmiWingHFDataset(EmmiWingDataset):
    """Emmi-Wing dataset loaded from the HuggingFace subset.

    Uses the 248-case evaluation scan subset with its own train/val/test splits.
    The dataset can be auto-downloaded from HuggingFace using :func:`download`.
    """

    def __init__(self, dataset_config: StandardDatasetConfig):
        super().__init__(dataset_config)
        logger.info(
            f"Using Emmi-Wing HF subset ({len(self.design_ids)} samples). "
            "Results are not comparable to models trained on the full dataset."
        )

    @property
    def get_dataset_splits(self) -> DatasetSplitIDs:
        return WingHFSplitIDs()

    @property
    def supported_splits(self) -> set[str]:
        return {"train", "test", "val"}

    @staticmethod
    def download(local_dir: str) -> str:
        """Download and extract the HF subset to a local directory.

        Downloads ``scans.zip`` from HuggingFace, extracts the nested ``run_N.zip`` archives into ``<local_dir>/run_N/``
        directories, and cleans up the zip files.

        Args:
            local_dir: Destination directory.

        Returns:
            Path to the extracted dataset root.
        """
        local_path = Path(local_dir)

        # Check if already extracted:
        existing_runs = list(local_path.glob("run_*"))
        if len(existing_runs) >= 248:
            logger.info(f"Dataset already extracted at {local_path} ({len(existing_runs)} runs)")
            return str(local_path)

        logger.info(f"Downloading Emmi-Wing HF subset to {local_path}...")
        local_path.mkdir(parents=True, exist_ok=True)

        scans_zip = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="scans.zip",
            repo_type="dataset",
            local_dir=str(local_path),
        )

        logger.info("Extracting scans.zip...")
        with zipfile.ZipFile(scans_zip, "r") as outer:
            outer.extractall(local_path)

        # scans.zip may extract into a subdirectory (e.g. "scans_with_density_zipped/")
        # Find wherever the run_*.zip files ended up.
        run_zips = sorted(local_path.rglob("run_*.zip"))
        if not run_zips:
            raise FileNotFoundError(f"No run_*.zip files found after extracting {scans_zip}")

        logger.info(f"Extracting {len(run_zips)} run archives...")
        for run_zip in run_zips:
            run_dir = local_path / run_zip.stem
            if not run_dir.exists():
                with zipfile.ZipFile(run_zip, "r") as inner:
                    inner.extractall(run_dir)
            run_zip.unlink()

        # Clean up the intermediate directory and outer zip
        for subdir in local_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("run_"):
                subdir.rmdir()  # remove empty intermediate directory
        Path(scans_zip).unlink(missing_ok=True)

        n_runs = len(list(local_path.glob("run_*")))
        logger.info(f"Extracted {n_runs} runs to {local_path}")
        return str(local_path)
