#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pydantic import ConfigDict, Field
from torch import Tensor

from noether.core.factory import DatasetFactory
from noether.core.schemas.dataset import DatasetBaseConfig, StandardDatasetConfig
from noether.data import Dataset


class LatentDatasetConfig(DatasetBaseConfig):
    """Config for the wrapped latent-diffusion dataset.

    The dataset combines two sources:

    * ``latent_root``: directory containing the per-sample ``.pt`` files
      written by :func:`extract_latents` (latents, supernode positions,
      optional super positions).
    * ``cfd_dataset_config``: full :class:`StandardDatasetConfig` for the
      original CFD dataset (e.g. DrivAerML). The wrapper instantiates this
      dataset internally and re-exposes its raw position/field
      ``getitem_*`` methods so the standard CFD pipeline can re-derive
      anchor positions, geometry tokens, and ground-truth fields at
      consume time. Inheriting the config (rather than just a path) keeps
      normalizer specs and split layout in one place.
    """

    model_config = ConfigDict(extra="forbid")

    latent_root: str
    """Directory holding the extracted latent ``.pt`` files (one per sample,
    grouped by split sub-directory)."""

    cfd_dataset_config: StandardDatasetConfig
    """Full config for the underlying CFD dataset. Provides ``getitem_*``
    accessors and normalizers; its own ``pipeline`` is unused — the latent
    dataset's pipeline runs the AB-UPT sample processors on the CFD raw
    data and adds a collator for the latent items."""

    split: str = Field("train")


class LatentDataset(Dataset):
    """Loads pre-extracted latent tokens paired with the original CFD sample.

    Each ``.pt`` file under ``latent_root/<split>/`` carries only the AE
    encoder outputs (``latents``, ``supernode_positions``, optional per-branch
    ``super_position_*``). Everything else needed by the downstream callbacks
    (anchor positions, geometry tokens, ground-truth fields) is re-derived at
    runtime by delegating to a wrapped CFD dataset built from
    ``cfd_dataset_config``: the latent dataset re-exposes the CFD dataset's
    raw position/field ``getitem_*`` methods so the standard AB-UPT pipeline
    can run its anchor-sampling and target-renaming sample processors on
    them, and the collator stage just stacks the resulting per-sample dict
    plus the latent items.

    Normalizers are inherited from the wrapped dataset so callbacks that
    denormalize predictions via ``data_container.get_dataset(...)`` keep
    working unchanged.
    """

    def __init__(self, dataset_config: LatentDatasetConfig, **kwargs: Any):
        super().__init__(dataset_config=dataset_config, **kwargs)

        latent_dir = Path(dataset_config.latent_root) / dataset_config.split
        self.files = sorted(latent_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"no .pt files found in {latent_dir}")

        cfd_cfg = dataset_config.cfd_dataset_config.model_copy(deep=True)
        cfd_cfg.split = dataset_config.split
        # The CFD dataset's own pipeline is unused — the latent dataset's
        # pipeline re-runs the same sample processors. Drop it so the wrapped
        # dataset stays a pure ``getitem_*`` source.
        cfd_cfg.pipeline = None
        self._cfd_dataset = DatasetFactory().create(cfd_cfg)

        # Inherit normalizers from the wrapped dataset (so
        # ``self.denormalize(key, ...)`` mirrors the CFD dataset's behaviour).
        self.normalizers = self._cfd_dataset.normalizers

        if len(self._cfd_dataset) != len(self.files):
            self.logger.warning(
                f"latent file count ({len(self.files)}) does not match CFD dataset length "
                f"({len(self._cfd_dataset)}); falling back to the latent count for __len__"
            )

    def __len__(self) -> int:
        return len(self.files)

    @property
    def cfd_dataset(self) -> Dataset:
        """The wrapped CFD dataset (raw positions/fields source)."""
        return self._cfd_dataset

    def _load_latent(self, idx: int) -> dict[str, Any]:
        return torch.load(self.files[idx], weights_only=True)

    def pre_getitem(self, idx: int) -> dict[str, Any]:
        """Cache the latent payload + raw CFD sample so per-field ``getitem_*`` calls are cheap."""
        return {
            "latent": self._load_latent(idx),
            "cfd": self._cfd_dataset[idx],
        }

    # --- latent-side accessors (read from the .pt payload) ----------------

    def getitem_latents(self, idx: int, **pre: Any) -> Tensor:
        """``(n_latent_tokens, latent_dim)`` AE-encoded latent."""
        return pre["latent"]["latents"]

    def getitem_supernode_positions(self, idx: int, **pre: Any) -> Tensor:
        """``(n_latent_tokens, 3)`` per-token geometry conditioning positions."""
        return pre["latent"]["supernode_positions"]

    def getitem_super_position_surface(self, idx: int, **pre: Any) -> Tensor | None:
        """``(K_surf, 3)`` super-token positions for the sampled bottleneck (else ``None``)."""
        return pre["latent"].get("super_position_surface")

    def getitem_super_position_volume(self, idx: int, **pre: Any) -> Tensor | None:
        """``(K_vol, 3)`` super-token positions for the sampled bottleneck (else ``None``)."""
        return pre["latent"].get("super_position_volume")

    # --- CFD-side accessors (delegate to the wrapped dataset) -------------

    def _cfd(self, key: str, pre: dict[str, Any]) -> Tensor:
        return pre["cfd"][key]

    def getitem_surface_position(self, idx: int, **pre: Any) -> Tensor:
        return self._cfd("surface_position", pre)

    def getitem_volume_position(self, idx: int, **pre: Any) -> Tensor:
        return self._cfd("volume_position", pre)

    def getitem_surface_pressure(self, idx: int, **pre: Any) -> Tensor:
        return self._cfd("surface_pressure", pre)

    def getitem_surface_friction(self, idx: int, **pre: Any) -> Tensor:
        return self._cfd("surface_friction", pre)

    def getitem_volume_pressure(self, idx: int, **pre: Any) -> Tensor:
        return self._cfd("volume_pressure", pre)

    def getitem_volume_velocity(self, idx: int, **pre: Any) -> Tensor:
        return self._cfd("volume_velocity", pre)

    def getitem_volume_vorticity(self, idx: int, **pre: Any) -> Tensor:
        return self._cfd("volume_vorticity", pre)
