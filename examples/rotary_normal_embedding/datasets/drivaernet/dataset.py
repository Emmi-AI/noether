#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import torch

from noether.core.schemas.dataset import DatasetBaseConfig
from noether.core.schemas.filemap import FileMap
from noether.core.utils.common.path import validate_path
from noether.data.datasets.cfd.dataset import AeroDataset

VALID_CATEGORIES = {
    "F_S_WWS_WM",
    "N_S_WWS_WM",
    "F_S_WWC_WM",
    "N_S_WWC_WM",
    "N_S_WW_WM",
    "E_S_WW_WM",
    "E_S_WWC_WM",
    "F_D_WM_WW",
}


class DrivAerNetDataset(AeroDataset):
    """Dataset implementation for DrivAerNet and DrivAerNet++ dataset."""

    FILEMAP = FileMap(
        surface_position="surface_position.pt",
        surface_pressure="surface_p.pt",
        surface_friction="surface_wallShearStress.pt",
        surface_normals="surface_Normals.pt",
        volume_position="volume_position.pt",
        volume_pressure="volume_p.pt",
        volume_velocity="volume_U.pt",
        volume_friction="volume_wallShearStress.pt",
        volume_vorticity="volume_vorticity.pt",
    )

    def __init__(self, dataset_config: DatasetBaseConfig):
        super().__init__(dataset_config=dataset_config, filemap=self.FILEMAP)

        self.source_root = validate_path(dataset_config.root)  # type: ignore[arg-type, attr-defined]
        if not self.source_root.exists():
            raise ValueError(f"Root directory '{self.source_root.as_posix()}' doesn't exist")

        datasplits = self._load_all_datasplits()

        if dataset_config.filter_categories is not None:  # type: ignore[attr-defined]
            for category in dataset_config.filter_categories:  # type: ignore[attr-defined]
                if category not in VALID_CATEGORIES:
                    raise ValueError(f"Invalid category: {category}. Valid categories: {VALID_CATEGORIES}")
            datasplits = {
                k: [id for id in v if "_".join(id.split("_")[:-1]) in dataset_config.filter_categories]  # type: ignore[attr-defined]
                for k, v in datasplits.items()
            }
            assert all(len(v) > 0 for v in datasplits.values()), "No samples left after filtering"

        # Parse the split parameter and extract any subset specification
        self.split, subset_indices = self._parse_split_subset(dataset_config.split)  # type: ignore[attr-defined]

        if self.split not in datasplits:
            raise ValueError(f"Unknown split '{self.split}'. Available splits: {list(datasplits.keys())}")

        # Apply subsetting if specified, otherwise use the full split
        all_design_ids = datasplits[self.split]
        self.design_ids = [all_design_ids[i] for i in subset_indices] if subset_indices else all_design_ids

    def _parse_split_subset(self, split_spec: str) -> tuple[str, list[int] | range | None]:
        """Parse a split specification with optional subset info in brackets.

        Examples:
            "train" -> ("train", None)  # No subset, use full split
            "train[0]" -> ("train", [0])  # Single sample
            "test[:50]" -> ("test", range(0, 50))  # First 50 samples

        Returns:
            Tuple of (split_name, subset_indices) where subset_indices is:
            - None: when no brackets are used, indicating full split should be used
            - List of indices or range object: when brackets are used
        """
        # No brackets means no subset specification - return None for subset_indices
        if not ("[" in split_spec and split_spec.endswith("]")):
            return split_spec, None

        # Get split name and subset expression
        split_name, subset_expr = split_spec.split("[", 1)
        subset_expr = subset_expr.rstrip("]")

        try:
            # Single index case: "train[0]"
            if ":" not in subset_expr:
                return split_name, [int(subset_expr)]

            # Slice notation: convert "10:20" → range(10, 20)
            parts = [int(p) if p else None for p in subset_expr.split(":")]
            if len(parts) == 2:  # start:end format
                start = parts[0] or 0
                end = parts[1] or 1000000
                return split_name, range(start, end)
            elif len(parts) == 3:  # start:end:step format
                start = parts[0] or 0
                end = parts[1] or 1000000
                step = parts[2] or 1
                return split_name, range(start, end, step)
            else:
                raise ValueError("Invalid slice format")
        except ValueError as err:
            raise ValueError(f"Invalid subset expression: '{subset_expr}'") from err

    def __len__(self):
        return len(self.design_ids)

    def _load_all_datasplits(self):
        """Load design IDs for all splits (train, test, and validation)."""
        datasplits_path = self.source_root
        split_design_ids_paths = {
            "train": datasplits_path / "train_design_ids.txt",
            "test": datasplits_path / "test_design_ids.txt",
            "val": datasplits_path / "val_design_ids.txt",
        }
        # The first blacklist was created automatically while preprocessing, since a file has inconsistent meshes
        blacklist1_design_ids_path = datasplits_path / "blacklist.txt"
        # The second blacklist was created manually, since some designs are just not there in the data -.-
        blacklist2_design_ids_path = datasplits_path / "blacklist2.txt"

        if not blacklist1_design_ids_path.exists():
            raise ValueError(f"Blacklist file does not exist: {blacklist1_design_ids_path.as_posix()}")
        if not blacklist2_design_ids_path.exists():
            raise ValueError(f"Blacklist2 file does not exist: {blacklist2_design_ids_path.as_posix()}")
        for split, path in split_design_ids_paths.items():
            if not path.exists():
                raise ValueError(f"Datasplit file for {split} does not exist: {path.as_posix()}")

        with open(self.source_root / "blacklist.txt") as f:
            blacklist1_ids = {line.strip().split("/")[-1].split(".vtk")[0] for line in f}
        with open(self.source_root / "blacklist2.txt") as f:
            blacklist2_ids = {line.strip().split("/")[-1].split(".vtk")[0] for line in f}
        blacklist_ids = blacklist1_ids | blacklist2_ids

        split_ids = dict()
        for split, path in split_design_ids_paths.items():
            with open(path) as f:
                split_ids[split] = [line.strip() for line in f if line.strip() not in blacklist_ids]
        total_samples = sum(len(paths) for paths in split_ids.values())
        total_samples_in_splits = 8121  # from 8129 apparently, only 8121 are in their split
        assert total_samples == total_samples_in_splits - len(blacklist_ids)
        return split_ids

    def _load(self, idx: int, filename: str) -> torch.Tensor:
        """Load a tensor file from the dataset."""
        idx = idx % len(self.design_ids)
        sample_uri = self.source_root / self.design_ids[idx] / filename
        if not sample_uri.exists():
            raise FileNotFoundError(f"File not found: {sample_uri}")
        return torch.load(sample_uri, weights_only=True)  # type: ignore[no-any-return]
