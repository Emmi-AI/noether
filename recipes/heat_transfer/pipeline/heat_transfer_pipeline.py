#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Any

from noether.data.pipeline import MultiStagePipeline, SampleProcessor
from noether.data.pipeline.collators import DefaultCollator
from noether.data.pipeline.collators.concat_sparse_tensor import ConcatSparseTensorCollator
from noether.data.pipeline.collators.sparse_tensor_offset import SparseTensorOffsetCollator
from noether.data.pipeline.multistage import PipelineConfig
from noether.data.pipeline.sample_processors import (
    DuplicateKeysSampleProcessor,
    PointSamplingSampleProcessor,
    RenameKeysSampleProcessor,
)
from noether.data.pipeline.sample_processors.supernode_sampling import SupernodeSamplingSampleProcessor
from noether.data.schemas import ModelDataSpecs


class HeatTransferPipelineConfig(PipelineConfig):
    """Pipeline configuration for volume-only heat transfer datasets."""

    kind: str | None = "heat_transfer.pipeline.HeatTransferPipeline"

    num_volume_points: int
    """Number of volume points to sample as input for the model."""
    num_volume_anchor_points: int | None = None
    """Number of volume anchor points for AB-UPT. If None, all sampled volume points are used as anchors."""

    num_geometry_points: int = 0
    """Number of geometry points to sample. Set to 0 if no geometry encoder is used."""

    num_geometry_supernodes: int = 0
    """Number of geometry supernodes for AB-UPT. Set to 0 if no geometry encoder is used."""

    data_specs: ModelDataSpecs
    """Data specifications for the pipeline."""
    seed: int | None = None
    """Random seed for deterministic sampling during evaluation."""


class HeatTransferPipeline(MultiStagePipeline):
    """Pipeline for volume-only heat transfer datasets (e.g., SimshiftHeatsink).

    Handles point sampling, anchor position creation, target renaming, and conditioning
    passthrough for datasets that have only a volume domain.
    """

    def __init__(self, pipeline_config: HeatTransferPipelineConfig, **kwargs) -> None:
        self.num_volume_points = pipeline_config.num_volume_points
        self.num_volume_anchor_points = pipeline_config.num_volume_anchor_points or self.num_volume_points
        self.num_geometry_points = pipeline_config.num_geometry_points
        self.num_geometry_supernodes = pipeline_config.num_geometry_supernodes
        self.seed = pipeline_config.seed

        volume_spec = pipeline_config.data_specs.domains.get("volume")
        self.volume_targets = {f"volume_{k}" for k in volume_spec.output_dims.keys()} if volume_spec else set()
        self.conditioning_dims = pipeline_config.data_specs.conditioning_dims

        super().__init__(
            sample_processors=self._build_sample_processors(),
            collators=self._build_collators(),
            **kwargs,
        )

    def _build_sample_processors(self) -> list[SampleProcessor]:
        processors: list[SampleProcessor] = []

        # 1. Sample volume points (position + all target fields together)
        volume_items = {"volume_position"} | self.volume_targets

        if self.num_geometry_points > 0 and self.num_geometry_supernodes > 0:
            processors.extend(
                [
                    DuplicateKeysSampleProcessor(
                        key_map={"volume_position": "geometry_position"},
                    ),
                    PointSamplingSampleProcessor(
                        items={"geometry_position"},
                        num_points=self.num_geometry_points,
                        seed=None if self.seed is None else self.seed + 2,
                    ),
                    SupernodeSamplingSampleProcessor(
                        item="geometry_position",
                        num_supernodes=self.num_geometry_supernodes,
                        supernode_idx_key="geometry_supernode_idx",
                        seed=None if self.seed is None else self.seed + 2,
                    ),
                ]
            )

        processors.append(
            PointSamplingSampleProcessor(
                items=volume_items,
                num_points=self.num_volume_points,
                seed=self.seed,
            )
        )

        # 2. If using fewer anchor points than sampled points, subsample for anchors.
        #    Otherwise, all sampled points serve as anchors.
        if self.num_volume_anchor_points < self.num_volume_points:
            # Create a second subsample for anchor points
            processors.append(
                DuplicateKeysSampleProcessor(
                    key_map={item: f"_anchor_{item}" for item in volume_items},
                )
            )
            processors.append(
                PointSamplingSampleProcessor(
                    items={f"_anchor_{item}" for item in volume_items},
                    num_points=self.num_volume_anchor_points,
                    seed=None if self.seed is None else self.seed + 1,
                )
            )
            # Rename _anchor_volume_position -> volume_anchor_position
            processors.append(RenameKeysSampleProcessor(key_map={"_anchor_volume_position": "volume_anchor_position"}))
            # Rename _anchor_volume_{field} -> volume_{field} (overwrites full-size with anchor-size)
            processors.append(
                RenameKeysSampleProcessor(
                    key_map={f"_anchor_{t}": t for t in self.volume_targets},
                )
            )
        else:
            # All sampled points are anchors — just rename position
            processors.append(RenameKeysSampleProcessor(key_map={"volume_position": "volume_anchor_position"}))

        # 3. Create target copies (volume_velocity -> volume_velocity_target, etc.)
        processors.append(
            DuplicateKeysSampleProcessor(
                key_map={target: f"{target}_target" for target in self.volume_targets},
            )
        )

        return processors

    def _build_collators(self) -> list[Any]:
        collate_items = ["volume_anchor_position"]
        collate_items += [f"{t}_target" for t in self.volume_targets]
        if self.conditioning_dims:
            collate_items += list(self.conditioning_dims.keys())
        collators: list[Any] = [DefaultCollator(items=collate_items)]

        if self.num_geometry_supernodes:
            # if we have geometry supernodes, we have to turn the geometry positions into a sparse tensor with batch indices.
            collators.extend(
                [
                    ConcatSparseTensorCollator(
                        items=["geometry_position"],
                        create_batch_idx=True,
                        batch_idx_key="geometry_batch_idx",
                    ),
                    SparseTensorOffsetCollator(
                        item="geometry_supernode_idx",
                        offset_key="geometry_position",
                    ),
                ]
            )
        return collators
