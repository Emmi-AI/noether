#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch

from noether.core.schemas.dataset import DatasetBaseConfig
from noether.data import Dataset, with_normalizers
from noether.data.datasets.cfd.caeml.filemap import FileMap


class AeroDataset(Dataset):
    """Dataset implementation for aerodynamic datasets with volume and surface fields.

    This unified dataset class provides an interface for aerodynamics dataset with volume and surface fields.
    The dataset behavior such as the dataset choice, train/val/test split IDs, etc.
    is configured through constructor parameters, allowing for easy extension to new datasets.

    Args:
        dataset_config: Configuration for the dataset.
    """

    def __init__(self, dataset_config: DatasetBaseConfig, filemap: FileMap) -> None:
        super().__init__(dataset_config=dataset_config)
        self.filemap = filemap

    def __len__(self):
        raise NotImplementedError

    def _load_from_disk(self, idx: int, filename: str) -> torch.Tensor:
        """
        Method to load data from disk. Must be implemented by subclasses (i.e., specific datasets).
        """
        raise NotImplementedError

    def _load(self, idx: int, filename: str) -> torch.Tensor:
        return self._load_from_disk(idx=idx, filename=filename)

    @with_normalizers("surface_position")
    def getitem_surface_position(self, idx: int) -> torch.Tensor:
        """Retrieves surface positions (num_surface_points, 3)"""
        return self._load(idx=idx, filename=self.filemap.surface_position)  # type: ignore[arg-type]

    @with_normalizers("surface_pressure")
    def getitem_surface_pressure(self, idx: int) -> torch.Tensor:
        """Retrieves surface pressures (num_surface_points, 1)"""
        return self._load(idx=idx, filename=self.filemap.surface_pressure).unsqueeze(1)  # type: ignore[arg-type]

    @with_normalizers("surface_friction")
    def getitem_surface_friction(self, idx: int) -> torch.Tensor:
        """Retrieves surface friction (=wallshearstress) (num_surface_points, 3)"""
        return self._load(idx=idx, filename=self.filemap.surface_friction)  # type: ignore[arg-type]

    @with_normalizers("volume_position")
    def getitem_volume_position(self, idx: int) -> torch.Tensor:
        """Retrieves volume position (num_volume_points, 3)"""
        return self._load(idx=idx, filename=self.filemap.volume_position)  # type: ignore[arg-type]

    @with_normalizers("volume_pressure")
    def getitem_volume_pressure(self, idx: int) -> torch.Tensor:
        """Retrieves volume pressures (num_volume_points, 1)"""
        return self._load(idx=idx, filename=self.filemap.volume_pressure).unsqueeze(1)  # type: ignore[arg-type]

    @with_normalizers("volume_velocity")
    def getitem_volume_velocity(self, idx: int) -> torch.Tensor:
        """Retrieves volume velocity (num_volume_points, 3)"""
        return self._load(idx=idx, filename=self.filemap.volume_velocity)  # type: ignore[arg-type]

    @with_normalizers("volume_vorticity")
    def getitem_volume_vorticity(self, idx: int) -> torch.Tensor:
        """Retrieves volume vorticity (num_volume_points, 3)"""
        return self._load(idx=idx, filename=self.filemap.volume_vorticity)  # type: ignore[arg-type]

    @with_normalizers("volume_sdf")
    def getitem_volume_sdf(self, idx: int) -> torch.Tensor:
        """
        Retrieve signed distance field at volume points.

        The SDF is computed with respect to the car body surface.

        Args:
            idx: Sample index

        Returns:
            Tensor of shape (num_volume_points, 1) containing SDF values
        """
        return self._load(idx=idx, filename=self.filemap.volume_distance_to_surface).unsqueeze(1)  # type: ignore[arg-type]

    def getitem_volume_normals(self, idx: int) -> torch.Tensor:
        """
        Retrieve normal vectors at volume points.

        Note: Volume normals are already normalized (unit vectors) and point towards the car body.

        Args:
            idx: Sample index

        Returns:
            Tensor of shape (num_volume_points, 3) containing unit normal vectors
        """
        return self._load(idx=idx, filename=self.filemap.volume_normals)  # type: ignore[arg-type]

    def getitem_surface_normals(self, idx: int) -> torch.Tensor:
        """
        Retrieve surface normal vectors.

        Note: Surface normals are already normalized (unit vectors).

        Args:
            idx: Sample index

        Returns:
            Tensor of shape (num_surface_points, 3) containing unit normal vectors
        """
        return self._load(idx=idx, filename=self.filemap.surface_normals)  # type: ignore[arg-type]
