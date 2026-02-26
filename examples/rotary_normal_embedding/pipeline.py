#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.data.pipeline import SampleProcessor
from noether.data.pipeline.collators import (
    ConcatSparseTensorCollator,
    DefaultCollator,
    SparseTensorOffsetCollator,
)
from noether.data.pipeline.sample_processors import (
    DuplicateKeysSampleProcessor,
    PointSamplingSampleProcessor,
    RenameKeysSampleProcessor,
    SupernodeSamplingSampleProcessor,
)
from tutorial.pipeline.multistage_pipelines.aero_multistage import (
    AeroMultistagePipeline as BaseAeroMultistagePipeline,
)
from tutorial.pipeline.multistage_pipelines.aero_multistage import DataKeys, _split_by_underscore, _split_three_or_none
from tutorial.pipeline.sample_processors import AnchorPointSamplingSampleProcessor


class AeroMultistagePipeline(BaseAeroMultistagePipeline):
    """
    Extended pipeline for CFD AeroMultistagePipeline with Rotary Normal Embedding (RoNE) support (i.e., load surface normals for the surface geometry).

    This extends the base AeroMultistagePipeline to handle surface normals alongside positions,
    which are used for normal-aware positional encoding in the model.

    Key extensions:
    - Includes geometry_normals in sparse tensor collation
    - Duplicates surface_normals to geometry_normals for geometry branch
    - Samples normals along with positions in anchor point sampling
    - Adds surface_anchor_normals to collator items
    """

    def _build_collator_pipeline(self) -> list:
        """
        Build the collators with support for geometry normals.

        Extended from base class to include geometry_normals in sparse tensor collation
        when geometry supernodes are used.
        """
        collators = []
        collators.extend([DefaultCollator(items=self.default_collator_items)])

        if self.num_supernodes > 0:
            # Same as base class
            collators.extend(
                [
                    ConcatSparseTensorCollator(
                        items=["surface_position"],
                        create_batch_idx=True,
                        batch_idx_key="surface_position_batch_idx",
                    ),
                    SparseTensorOffsetCollator(
                        item="surface_position_supernode_idx",
                        offset_key="surface_position",
                    ),
                ]
            )

        if self.num_geometry_supernodes:
            # EXTENDED: Include geometry_normals for RoNE
            collators.extend(
                [
                    ConcatSparseTensorCollator(
                        items=["geometry_position", "geometry_normals"],
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

    def _get_anchor_point_sampling_sample_processor(self) -> list[SampleProcessor]:
        """
        Get the anchor point sampling sample processor with normal support.

        Extended from base class to:
        1. Add surface_anchor_normals to default collator items
        2. Duplicate surface_normals to geometry_normals
        3. Sample geometry_normals along with geometry_position
        4. Include surface_normals in anchor point sampling
        """
        if self.num_volume_anchor_points > 0 and self.num_surface_anchor_points > 0:
            # EXTENDED: Add surface_anchor_normals to collator items
            self.default_collator_items += [
                "surface_anchor_position",
                "volume_anchor_position",
                "surface_anchor_normals",
            ]
            return [
                DuplicateKeysSampleProcessor(key_map={"surface_position": "geometry_position"}),
                # EXTENDED: Duplicate surface normals for geometry branch
                DuplicateKeysSampleProcessor(key_map={"surface_normals": "geometry_normals"}),
                # EXTENDED: Sample both geometry position and normals
                PointSamplingSampleProcessor(
                    items={"geometry_position", "geometry_normals"},
                    num_points=self.num_geometry_points,
                    seed=None if self.seed is None else self.seed + 1,
                ),
                # Same as base class - but uses the sampled geometry_position above
                SupernodeSamplingSampleProcessor(
                    item="geometry_position",
                    num_supernodes=self.num_geometry_supernodes,
                    supernode_idx_key="geometry_supernode_idx",
                    seed=None if self.seed is None else self.seed + 2,
                ),
                # EXTENDED: Include surface_normals in surface anchor sampling
                AnchorPointSamplingSampleProcessor(
                    items={"surface_position", "surface_normals"} | set(self.surface_targets),
                    num_points=self.num_surface_anchor_points,
                    keep_queries=self.use_query_positions,
                    to_prefix_and_postfix=_split_by_underscore,
                    to_prefix_midfix_postfix=_split_three_or_none,
                    seed=None if self.seed is None else self.seed + 3,
                ),
                # Same as base class - volume anchors don't need normals
                AnchorPointSamplingSampleProcessor(
                    items={"volume_position"} | set(self.volume_targets),
                    num_points=self.num_volume_anchor_points,
                    keep_queries=self.use_query_positions,
                    to_prefix_and_postfix=_split_by_underscore,
                    to_prefix_midfix_postfix=_split_three_or_none,
                    seed=None if self.seed is None else self.seed + 4,
                ),
                RenameKeysSampleProcessor(key_map={DataKeys.as_anchor(key): key for key in self.volume_targets}),
                RenameKeysSampleProcessor(key_map={DataKeys.as_anchor(key): key for key in self.surface_targets}),
            ]
        else:
            raise ValueError(
                "Anchor point sampling requires both num_volume_anchor_points and num_surface_anchor_points to be greater than 0."
            )
