#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.data.pipeline import MultiStagePipeline, SampleProcessor
from noether.data.pipeline.collators import (
    ConcatSparseTensorCollator,
    DefaultCollator,
    SparseTensorOffsetCollator,
)
from noether.data.pipeline.sample_processors import (
    ConcatTensorSampleProcessor,
    DefaultTensorSampleProcessor,
    DuplicateKeysSampleProcessor,
    MomentNormalizationSampleProcessor,
    PointSamplingSampleProcessor,
    RenameKeysSampleProcessor,
    SupernodeSamplingSampleProcessor,
)
from tutorial.pipeline.sample_processors import (
    AnchorPointSamplingSampleProcessor,
)
from tutorial.schemas.pipelines.aero_pipeline_config import AeroCFDPipelineConfig


def _split_by_underscore(item: str) -> list[str]:
    return item.split("_")


def _split_three_or_none(item: str) -> list[str | None]:
    parts = item.split("_")
    return parts if len(parts) == 3 else [None] * 3


class DataKeys:
    """A central repository for data dictionary keys."""

    # Base Positions
    SURFACE_POS = "surface_position"
    VOLUME_POS = "volume_position"
    GEOMETRY_POS = "geometry_position"
    SURFACE_FEATURES = "surface_features"
    VOLUME_FEATURES = "volume_features"
    SURFACE_QUERY_FEATURES = "surface_query_features"
    VOLUME_QUERY_FEATURES = "volume_query_features"

    # Geometry
    GEOMETRY_BATCH_IDX = "geometry_batch_idx"
    GEOMETRY_SUPERNODE_IDX = "geometry_supernode_idx"

    @staticmethod
    def as_query(key: str) -> str:
        """Converts a standard key to its 'query' equivalent."""
        parts = key.split("_")
        assert len(parts) == 2, "Key must be in the format '<type>_<descriptor>'"
        return f"{parts[0]}_query_{parts[1]}"

    @staticmethod
    def as_target(key: str) -> str:
        """Converts a standard key to its 'target' equivalent."""
        return f"{key}_target"

    @staticmethod
    def as_anchor(key: str) -> str:
        """Converts a standard key to its 'anchor' equivalent."""
        parts = key.split("_")
        assert len(parts) == 2, "Key must be in the format '<type>_<descriptor>'"
        return f"{parts[0]}_anchor_{parts[1]}"


class AeroMultistagePipeline(MultiStagePipeline):
    """
    A pipeline for the the CFD AeroDynamics dataset AhmedML, DrivAerML, and ShapeNet-Car that handles multi-stage data processing.
    """

    @property
    def has_query_points(self) -> bool:
        """Check if any query points are specified."""
        return self.num_surface_queries + self.num_volume_queries > 0

    @property
    def use_anchor_points(self) -> bool:
        """Check if anchor points are used instead of standard sampling."""
        return self.num_volume_anchor_points > 0 and self.num_surface_anchor_points > 0

    def __init__(
        self,
        pipeline_config: AeroCFDPipelineConfig,
        **kwargs,
    ):
        """

        Args:
            pipeline_config: Configuration for the ShapeNet multi-stage pipeline.
        """

        self.dataset_statistics = pipeline_config.dataset_statistics
        self.seed = pipeline_config.seed

        # Number of points and queries for point sampling
        self.num_surface_points = pipeline_config.num_surface_points
        self.num_volume_points = pipeline_config.num_volume_points
        self.num_surface_queries = pipeline_config.num_surface_queries
        self.num_volume_queries = pipeline_config.num_volume_queries
        self.sample_query_points = pipeline_config.sample_query_points

        # UPT specific parameters
        self.num_supernodes = pipeline_config.num_supernodes

        # AB-UPT specific parameters
        self.num_volume_anchor_points = pipeline_config.num_volume_anchor_points
        self.num_surface_anchor_points = pipeline_config.num_surface_anchor_points
        self.num_geometry_points = pipeline_config.num_geometry_points
        self.num_geometry_supernodes = pipeline_config.num_geometry_supernodes
        self.use_query_positions = False

        self.use_physics_features = (
            pipeline_config.use_physics_features
        )  # Whether to use physics features (i.e., SDF, normals, etc.) as input to the model.

        self.surface_features = pipeline_config.data_specs.surface_features
        self.volume_features = pipeline_config.data_specs.volume_features
        self.surface_targets = pipeline_config.data_specs.surface_targets
        self.volume_targets = pipeline_config.data_specs.volume_targets
        self.conditioning_dims = pipeline_config.data_specs.conditioning_dims

        self._define_items_keys()

        super().__init__(
            sample_processors=self._build_sample_processor_pipeline(),
            collators=self._build_collator_pipeline(),
            batch_processors=self._build_batch_processor_pipeline(),
            **kwargs,
        )

    def _define_items_keys(self) -> None:
        """
        When sampling input points and queries points, we have to tie certain items together.
        For example, the volume points needs to be sampled together with the matching targets and features.
        In this methods, we defined which dataset modes are sampled together for the volume and surface points.
        Next to that, we also define the query items, which are the same as the sampling items, but with a "query" prefix.
        We also define the default pipeline items, which are the items that are always present in the pipeline.
        """
        self.volume_sampling_items = (
            {
                DataKeys.VOLUME_POS,
            }
            | self.volume_targets
            if self.num_volume_points > 0
            else set()
        )

        self.volume_sampling_items |= self.volume_features if self.use_physics_features else set()
        self.surface_sampling_items = (
            {
                DataKeys.SURFACE_POS,
            }
            | self.surface_targets
            if self.num_surface_points > 0
            else set()
        )
        self.surface_sampling_items |= self.surface_features if self.use_physics_features else set()

        self.volume_query_items = {DataKeys.as_query(item) for item in self.volume_sampling_items}

        self.surface_query_items = {DataKeys.as_query(item) for item in self.surface_sampling_items}

        # By default we collate the input positions and the surface mask of the input points.
        self.default_collator_items = (
            [
                DataKeys.VOLUME_POS,
            ]
            if self.num_volume_anchor_points == 0
            else []
        )

        if self.num_volume_anchor_points == 0 or self.num_surface_anchor_points == 0:
            self.default_collator_items += (
                [DataKeys.SURFACE_POS]
                if self.num_supernodes == 0
                else ["surface_query_position", "volume_query_position"]
            )

        # next to that we also collate the physics features, which are the concatenation of the surface and volume features. The targets are also included.
        self.default_collator_items += [DataKeys.as_target(item) for item in self.surface_targets | self.volume_targets]
        self.default_collator_items += (
            [DataKeys.VOLUME_FEATURES, DataKeys.SURFACE_FEATURES] if self.use_physics_features else []
        )
        self.default_collator_items += (
            [DataKeys.SURFACE_QUERY_FEATURES, DataKeys.VOLUME_QUERY_FEATURES]
            if self.has_query_points and self.use_physics_features
            else []
        )
        self.default_collator_items += self.conditioning_dims.keys() if self.conditioning_dims else []

    def _build_sample_processor_pipeline(self) -> list[SampleProcessor]:
        """
        Build the sample processor pipeline.
        """
        sample_processors = []
        # Some tensors are always present with the same value (i.e., the SDF value on the surface is always 0.0), we first create get the sample processors for these tensors.
        sample_processors.extend(self._get_default_tensors_sample_processors())
        # We need to normalize the input tensors individually, so we create the normalizers for the surface and volume tensors.
        sample_processors.extend(self._get_normalizer_sample_processors())
        sample_processors.extend(self._get_point_sampling_sample_processors())
        # certain tensors need to be concatenated to create the input tensors for the model
        sample_processors.extend(self._get_concatate_tensors_sample_processors())
        # We need to rename the target tensors to match the model output keys.
        sample_processors.extend(self._get_target_renaming_sample_processors())

        return sample_processors

    def _build_collator_pipeline(self) -> list:
        """
        Build the collators. Collators define how the  are combined into a batch.
        """

        collators = []
        collators.extend(
            [
                DefaultCollator(
                    items=self.default_collator_items,
                )
            ]
        )
        if self.num_supernodes > 0:
            # if we have supernodes, we have to turn the surface positions into a sparse tensor with batch indices.
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

    def _build_batch_processor_pipeline(self) -> list:
        """
        Build the batch processors.
        """
        return []

    def _get_normalizer_sample_processors(self) -> list[SampleProcessor]:
        """We get different sample processors for the surface and volume tensors. The input coordinates (i.e., positions) are also normalized in a different way."""
        return [
            *self._get_surface_normalizers_sample_processors(),
        ]

    def _get_point_sampling_sample_processors(self) -> list[SampleProcessor]:
        """
        We now get the point sampling sample processors, which sample the surface and volume points in different ways.
        If we use anchor points, we sample the anchor points instead of the standard surface and volume points.
        For all the other baselines, we first sample query points (if any) and then sample the input points.
        """

        if self.num_volume_anchor_points > 0 and self.num_surface_anchor_points > 0:
            return [*self._get_anchor_point_sampling_sample_processor()]
        else:
            return [*self._get_query_sampling_sample_processor(), *self._get_input_sampling_sample_processor()]

    def _get_default_tensors_sample_processors(self) -> list[SampleProcessor]:
        """Some tensors are always present in the dataset with a default value, so we create a default tensor to create it"""
        if self.use_physics_features is False:
            return []
        return [
            # the SDF of the surface is always 0.0, so we create a default tensor for it.
            DefaultTensorSampleProcessor(
                item_key_name="surface_sdf",
                matching_item_key="surface_position",
                feature_dim=1,
                default_value=0.0,
            )
        ]

    def _get_surface_normalizers_sample_processors(self) -> list[SampleProcessor]:
        """
        Get the normalizer for surface quantities.
        """
        if self.use_physics_features is False:
            return []
        return [
            MomentNormalizationSampleProcessor(
                item="surface_sdf",
                mean=self.dataset_statistics.volume_sdf_mean,
                std=self.dataset_statistics.volume_sdf_std,
            ),
        ]

    def _get_input_sampling_sample_processor(self) -> list[SampleProcessor]:
        """
        Get the point sampling sample processor.
        """
        assert self.num_volume_points + self.num_surface_points > 0, (
            "At least one of num_volume_points or num_surface_points must be greater than 0."
        )
        sample_processors = [
            PointSamplingSampleProcessor(
                items=self.volume_sampling_items,
                num_points=self.num_volume_points,
                seed=self.seed,
            ),
            PointSamplingSampleProcessor(
                items=self.surface_sampling_items,
                num_points=self.num_surface_points,
                seed=self.seed,
            ),
        ]
        if self.has_query_points and not self.sample_query_points:
            # we use the same sampling items for the query points as for the surface and volume points
            sample_processors.extend(
                [
                    DuplicateKeysSampleProcessor(key_map={item: DataKeys.as_query(item)})
                    for item in self.volume_sampling_items | self.surface_sampling_items
                ]
            )
        if self.num_supernodes > 0:
            sample_processors.append(
                SupernodeSamplingSampleProcessor(
                    item="surface_position",
                    num_supernodes=self.num_supernodes,
                    supernode_idx_key="surface_position_supernode_idx",
                )
            )
        return sample_processors

    def _get_query_sampling_sample_processor(self) -> list[SampleProcessor]:
        """
        Get the query sampling sample processor.
        """
        if self.has_query_points and self.sample_query_points:
            # we first have to duplicate the keys for the query points, so that we can sample them separately
            quey_keymap = {
                item: DataKeys.as_query(item) for item in self.surface_sampling_items | self.volume_sampling_items
            }
            return [
                DuplicateKeysSampleProcessor(key_map=quey_keymap),
                PointSamplingSampleProcessor(
                    items=self.surface_query_items,
                    num_points=self.num_surface_queries,
                    seed=self.seed,
                ),
                PointSamplingSampleProcessor(
                    items=self.volume_query_items, num_points=self.num_volume_queries, seed=self.seed
                ),
            ]

        else:
            return []

    def _get_concatate_tensors_sample_processors(self) -> list[SampleProcessor]:
        """
        For most models, the input to the encoder, the query points, and hence the output targets are the concatenation of the surface and volume points.
        We concatenate the surface and volume positions, features, and physics features.
        """
        sample_processors = []
        if self.use_physics_features:
            sample_processors.extend(
                [
                    ConcatTensorSampleProcessor(
                        items=self.volume_features,
                        target_key="volume_features",
                        dim=1,
                    ),
                    ConcatTensorSampleProcessor(
                        items=self.surface_features,
                        target_key="surface_features",
                        dim=1,
                    ),
                ]
            )

        if self.has_query_points:
            # if we have query points, we also concatenate the query positions and features
            if self.use_physics_features:
                sample_processors.extend(
                    [
                        ConcatTensorSampleProcessor(
                            items={DataKeys.as_query(item) for item in self.volume_features},
                            target_key="volume_query_features",
                            dim=1,
                        ),
                        ConcatTensorSampleProcessor(
                            items={DataKeys.as_query(item) for item in self.surface_features},
                            target_key="surface_query_features",
                            dim=1,
                        ),
                    ]
                )

        return sample_processors

    def _get_target_renaming_sample_processors(self) -> list[SampleProcessor]:
        """The quantities we predict are the surface pressure and volume velocity, which are the targets of the model.
        We rename the surface pressure and volume velocity to match the model output keys.
        """
        if self.has_query_points:
            return [
                DuplicateKeysSampleProcessor(
                    key_map={DataKeys.as_query(target): DataKeys.as_target(target) for target in self.volume_targets}
                ),
                DuplicateKeysSampleProcessor(
                    key_map={DataKeys.as_query(target): DataKeys.as_target(target) for target in self.surface_targets}
                ),
            ]
        else:
            return [
                DuplicateKeysSampleProcessor(
                    key_map={target: DataKeys.as_target(target) for target in self.volume_targets}
                ),
                DuplicateKeysSampleProcessor(
                    key_map={target: DataKeys.as_target(target) for target in self.surface_targets}
                ),
            ]

    def _get_anchor_point_sampling_sample_processor(self) -> list[SampleProcessor]:
        """Get the anchor point sampling sample processor."""
        if self.num_volume_anchor_points > 0 and self.num_surface_anchor_points > 0:
            # make sure defa
            self.default_collator_items += [
                "surface_anchor_position",
                "volume_anchor_position",
            ]
            return [
                DuplicateKeysSampleProcessor(key_map={"surface_position": "geometry_position"}),
                PointSamplingSampleProcessor(
                    items={"geometry_position"},
                    num_points=self.num_geometry_points,
                    seed=None if self.seed is None else self.seed + 1,
                ),
                SupernodeSamplingSampleProcessor(
                    item="geometry_position",
                    num_supernodes=self.num_geometry_supernodes,
                    supernode_idx_key="geometry_supernode_idx",
                    seed=None if self.seed is None else self.seed + 2,
                ),
                # subsample surface data
                AnchorPointSamplingSampleProcessor(
                    items={"surface_position"} | set(self.surface_targets),
                    num_points=self.num_surface_anchor_points,
                    keep_queries=self.use_query_positions,
                    to_prefix_and_postfix=_split_by_underscore,
                    to_prefix_midfix_postfix=_split_three_or_none,
                    seed=None if self.seed is None else self.seed + 3,
                ),
                # subsample volume data
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
