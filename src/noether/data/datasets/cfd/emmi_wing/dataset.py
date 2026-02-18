#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging

import torch

from noether.core.schemas.dataset import CAEMLDatasetConfig, DatasetSplitIDs
from noether.core.schemas.filemap import FileMap
from noether.core.utils.common import validate_path
from noether.data.base.dataset import with_normalizers
from noether.data.datasets.cfd.dataset import AeroDataset
from noether.data.datasets.cfd.emmi_wing.split import WingParametricSplitIDs

logger = logging.getLogger(__name__)

WING_FILE_MAP = FileMap(
    surface_position="surface_position.pt",
    surface_pressure="surface_pressure.pt",
    surface_friction="surface_wall_shear_stress.pt",
    volume_position="volume_position.pt",
    volume_pressure="volume_pressure.pt",
    volume_velocity="volume_velocity.pt",
    volume_vorticity="volume_vorticity.pt",
    design_parameters="design_parameters.pt",
)


class EmmiWingDataset(AeroDataset):
    def __init__(self, dataset_config: CAEMLDatasetConfig):
        super().__init__(dataset_config=dataset_config, filemap=WING_FILE_MAP)

        self.split = dataset_config.split
        if self.split not in self.supported_splits:
            raise ValueError(f"Unsupported split '{self.split}'. Supported splits are: {self.supported_splits}")
        self.source_root = validate_path(dataset_config.root)  # type: ignore[arg-type]
        self._load_design_ids()
        logger.info(f"Initialized Emmi-Wing with {len(self.design_ids)} samples for split '{self.split}'")

    def _load_design_ids(self) -> None:
        """
        Load URIs for dataset samples based on the split.
        """
        self.design_ids = list(self.get_dataset_splits.model_dump()[self.split])

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            Number of simulation samples available
        """
        return len(self.design_ids)

    def _load(self, idx: int, filename: str) -> torch.Tensor:
        """
        Load a tensor from a specific file in a sample directory.

        Args:
            idx: Index of the sample to load (automatically wrapped with modulo)
            filename: Name of the file to load from the sample directory

        Returns:
            Loaded tensor from the specified file

        Raises:
            FileNotFoundError: If the requested file does not exist
            RuntimeError: If loading fails for any reason
        """

        sample_uri = self.source_root / f"run_{self.design_ids[idx]}" / filename
        try:
            return torch.load(sample_uri, weights_only=True)  # type: ignore[no-any-return]
        except Exception as e:
            raise RuntimeError(f"Failed to load {sample_uri}, {filename}: {e}") from e

    def sample_info(self, idx: int) -> dict[str, str | int | None]:
        """Get information about a sample such as its local path, run name, etc."""
        idx = idx % len(self.design_ids)
        design_id = self.design_ids[idx]
        run_name = f"run_{design_id}"
        sample_uri = self.source_root / run_name
        return {
            "sample_uri": sample_uri,
            "run_name": run_name,
            "run_number": design_id,
            "design_id": design_id,
            "split": self.split,
        }

    @property
    def get_dataset_splits(self) -> DatasetSplitIDs:
        return WingParametricSplitIDs()

    @property
    def supported_splits(self) -> set[str]:
        return {"train", "test", "val", "test_extrapol", "test_interpol", "train_subset"}

    def _load_design_parameters_map(self, idx: int) -> torch.Tensor | dict[str, torch.Tensor]:
        """Retrieves all design parameters as dictionary {<param_name>: <param_value>}.
        Assumes that the design parameters are stored as a dictionary in the file."""
        if self.filemap.design_parameters is None:
            raise ValueError("design_parameters not available for this dataset")
        return self._load(idx=idx, filename=self.filemap.design_parameters)

    @with_normalizers("geometry_design_parameters")
    def getitem_geometry_design_parameters(self, idx: int) -> torch.Tensor:
        """Retrieves geometry design parameters as a single tensor.

        Returns:
            torch.Tensor: Geometry design parameters tensor of shape (1, num_geometry_parameters)
        """

        params_dict = self._load_design_parameters_map(idx)

        if not isinstance(params_dict, dict):
            raise TypeError(f"Expected dict for parameters, got {type(params_dict)}")

        geometry_params = torch.tensor(
            [params_dict[name] for name in ["chord_root", "span", "taper_ratio", "sweep", "dihedral"]]
        )
        # unsqueeze for collator normalizer compatibility
        return geometry_params.unsqueeze(0)

    @with_normalizers("inflow_design_parameters")
    def getitem_inflow_design_parameters(self, idx: int) -> torch.Tensor:
        """Retrieves inflow design parameters as a single tensor.

        Returns:
            torch.Tensor: Inflow design parameters tensor of shape (1, num_inflow_parameters)
        """
        params_dict = self._load_design_parameters_map(idx)

        if not isinstance(params_dict, dict):
            raise TypeError(f"Expected dict for parameters, got {type(params_dict)}")

        inflow_params = torch.tensor([params_dict[name] for name in ["inflow_velocity", "aoa"]])
        # unsqueeze for collator normalizer compatibility
        return inflow_params.unsqueeze(0)
