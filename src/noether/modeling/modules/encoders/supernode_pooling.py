#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import einops
import torch
from torch import nn

from noether.core.schemas.modules.encoders import SupernodePoolingConfig
from noether.core.schemas.modules.layers import ContinuousSincosEmbeddingConfig, LinearProjectionConfig
from noether.modeling.functional.geometric import knn, radius, segment_csr
from noether.modeling.modules.activations import Activation
from noether.modeling.modules.layers import ContinuousSincosEmbed, LinearProjection


class SupernodePooling(nn.Module):
    """Supernode pooling layer.

    The permutation of the supernodes is preserved through the message passing (contrary to the (GP-)UPT code).
    Additionally, radius is used instead of radius_graph, which is more efficient.
    """

    def __init__(
        self,
        config: SupernodePoolingConfig,
    ):
        """Initialize the SupernodePooling.

        Args:
            config: Configuration for the SupernodePooling module.

        """
        super().__init__()

        self.radius = config.radius
        self.k = config.k
        self.max_degree = config.max_degree
        self.spool_pos_mode = config.spool_pos_mode
        self.readd_supernode_pos = config.readd_supernode_pos
        self.aggregation = config.aggregation
        self.input_features_dim = config.input_features_dim

        self.pos_embed = ContinuousSincosEmbed(
            config=ContinuousSincosEmbeddingConfig(
                hidden_dim=config.hidden_dim,
                input_dim=config.input_dim,
            )  # type: ignore[call-arg]
        )

        if self.input_features_dim is not None:
            self.feature_projection: LinearProjection | None = LinearProjection(
                config=LinearProjectionConfig(
                    input_dim=self.input_features_dim, output_dim=config.hidden_dim, init_weights=config.init_weights
                )  # type: ignore[call-arg]
            )
            if self.spool_pos_mode == "relpos":
                self.feature_down_projection: LinearProjection | None = LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=config.hidden_dim * 3, output_dim=config.hidden_dim, init_weights=config.init_weights
                    )  # type: ignore[call-arg]
                )
            else:
                self.feature_down_projection = LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=config.hidden_dim * 2, output_dim=config.hidden_dim, init_weights=config.init_weights
                    )  # type: ignore[call-arg,arg-type]
                )
        else:
            self.feature_projection = None
            self.feature_down_projection = None

        if config.spool_pos_mode == "abspos":
            message_input_dim = config.hidden_dim * 2
            self.rel_pos_embed: ContinuousSincosEmbed | None = None
        elif config.spool_pos_mode == "absrelpos":
            message_input_dim = config.hidden_dim * 3
            self.rel_pos_embed = ContinuousSincosEmbed(
                config=ContinuousSincosEmbeddingConfig(hidden_dim=config.hidden_dim, input_dim=config.input_dim + 1)  # type: ignore[call-arg]
            )
        elif config.spool_pos_mode == "relpos":
            message_input_dim = config.hidden_dim
            self.rel_pos_embed = ContinuousSincosEmbed(
                config=ContinuousSincosEmbeddingConfig(hidden_dim=config.hidden_dim, input_dim=config.input_dim + 1)  # type: ignore[call-arg]
            )
        else:
            raise NotImplementedError(
                "spool_pos_mode not implemented, needs to be one of 'abspos', 'relpos' or 'absrelpos'"
            )

        if config.message_mode == "mlp":
            self.message: nn.Sequential | LinearProjection | nn.Identity = nn.Sequential(
                LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=message_input_dim, output_dim=config.hidden_dim, init_weights=config.init_weights
                    )  # type: ignore[call-arg]
                ),
                Activation.GELU.value,
                LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=config.hidden_dim, output_dim=config.hidden_dim, init_weights=config.init_weights
                    )  # type: ignore[call-arg]
                ),
            )
        elif config.message_mode == "linear":
            self.message = LinearProjection(
                config=LinearProjectionConfig(
                    input_dim=message_input_dim, output_dim=config.hidden_dim, init_weights=config.init_weights
                )  # type: ignore[call-arg]
            )
        elif config.message_mode == "identity":
            self.message = nn.Identity()
        else:
            raise NotImplementedError("message_mode not implemented, needs to be one of 'mlp', 'linear' or 'identity'")
        if config.readd_supernode_pos:
            self.proj: LinearProjection | None = LinearProjection(
                config=LinearProjectionConfig(
                    input_dim=2 * config.hidden_dim, output_dim=config.hidden_dim, init_weights=config.init_weights
                )  # type: ignore[call-arg]
            )
        else:
            self.proj = None

        self.output_dim = config.hidden_dim

    def compute_src_and_dst_indices(
        self,
        input_pos: torch.Tensor,
        supernode_idx: torch.Tensor,
        batch_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the source and destination indices for the message passing to the supernodes.

        Args:
            input_pos: Sparse tensor with shape (batch_size * number of points, 3), representing the input geometries.
            supernode_idx: Indexes of the supernodes in the sparse tensor input_pos.
            batch_idx: 1D tensor, containing the batch index of each entry in input_pos. Default None.

        Returns:
            Tensor with src and destination indexes for the message passing into the supernodes.
        """

        # radius graph
        if batch_idx is None:
            batch_y = None
        else:
            batch_y = batch_idx[supernode_idx]
        if self.radius is not None:
            assert self.k is None
            edges = radius(
                x=input_pos,
                y=input_pos[supernode_idx],
                r=self.radius,
                max_num_neighbors=self.max_degree,
                batch_x=batch_idx,
                batch_y=batch_y,
            )
        elif self.k is not None:
            edges = knn(
                x=input_pos,
                y=input_pos[supernode_idx],
                k=self.k,
                batch_x=batch_idx,
                batch_y=batch_y,
            )
        else:
            raise NotImplementedError
        # remap dst indices
        dst_idx, src_idx = edges.unbind()
        dst_idx = supernode_idx[dst_idx]

        return src_idx, dst_idx

    def create_messages(
        self,
        input_pos: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        supernode_idx: torch.Tensor,
        input_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create messages for the message passing to the supernodes, based on different positional encoding
        representations.

        Args:
            input_pos: Tensor of shape (batch_size * number_of_points_per_sample, {2,3}), representing the point cloud
                representation of the input geometry.
            src_idx: Index of the source nodes from input_pos.
            dst_idx: Source index of the destination nodes from input_pos tensor. These indexes should be the matching
                supernode indexes.
            supernode_idx: Indexes of the node in input_pos that are considered supernodes.

        Raises:
            NotImplementedError: Raised if the mode is not implemented. Either "abspos", "relpos" or "absrelpos" are
                allowed.

        Returns:
            Tensor with messages for the message passing into the super nodes and the embedding coordinates of the
                supernodes.
        """

        # create message
        if self.spool_pos_mode == "abspos":
            x = self.pos_embed(input_pos)

            if self.input_features_dim is not None:
                if callable(self.feature_projection):
                    x = torch.concat([x, self.feature_projection(input_features)], dim=1)
                if callable(self.feature_projection):
                    x = self.feature_down_projection(x)
            else:
                assert input_features is None

            supernode_pos_embed = x[supernode_idx]
            x = torch.concat([x[src_idx], x[dst_idx]], dim=1)
        elif self.spool_pos_mode == "absrelpos":
            if self.input_features_dim is not None or input_features is not None and self.rel_pos_embed is not None:
                raise NotImplementedError
            src_pos = input_pos[src_idx]
            dst_pos = input_pos[dst_idx]
            dist = dst_pos - src_pos
            mag = dist.norm(dim=1).unsqueeze(-1)
            x = self.pos_embed(input_pos)
            supernode_pos_embed = x[supernode_idx]
            relemb = self.rel_pos_embed(torch.concat([dist, mag], dim=1))  # type: ignore[misc]
            x = torch.concat([x[src_idx], x[dst_idx], relemb], dim=1)
        elif self.spool_pos_mode == "relpos":
            src_pos = input_pos[src_idx]
            dst_pos = input_pos[dst_idx]
            dist = dst_pos - src_pos
            mag = dist.norm(dim=1).unsqueeze(-1)
            if self.input_features_dim is not None and input_features is not None and self.rel_pos_embed is not None:
                x = input_features.clone()
                if callable(self.feature_projection):
                    x = self.feature_projection(x)  # type: ignore[misc]
                x = torch.concat([self.rel_pos_embed(torch.concat([dist, mag], dim=1)), x[src_idx], x[dst_idx]], dim=1)  # type: ignore[misc]
                if callable(self.feature_down_projection):
                    x = self.feature_down_projection(x)
            else:
                assert input_features is None
                x = self.rel_pos_embed(torch.concat([dist, mag], dim=1))  # type: ignore[misc]
            supernode_pos_embed = self.pos_embed(input_pos[supernode_idx])
        else:
            raise NotImplementedError

        x = self.message(x)
        return x, supernode_pos_embed

    def accumulate_messages(
        self,
        x: torch.Tensor,
        dst_idx: torch.Tensor,
        supernode_idx: torch.Tensor,
        batch_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Method to accumulate the messages of neighbouring points into the supernodes.

        Args:
            x: Tensor containing the message representation of each neighbour representation.
            dst_idx: Index of the destination (i.e., supernode) where each message should go to.
            supernode_idx: Indexes of the supernode in the input point cloud.
            batch_idx: Batch index of the points in the sparse tensor.

        Returns:
            Tensor with the aggregated messages for each supernode.
        """
        # accumulate messages
        # indptr is a tensor of indices between which to aggregate
        # i.e. a tensor of [0, 2, 5] would result in [src[0] + src[1], src[2] + src[3] + src[4]]
        dst_indices, counts = dst_idx.unique_consecutive(return_counts=True)
        if not torch.all(dst_indices == supernode_idx):
            raise ValueError("dst_indices must match supernode_idx")

        # first index has to be 0
        # NOTE: padding for target indices that don't occour is not needed as self-loop is always present
        padded_counts = torch.zeros(len(counts) + 1, device=counts.device, dtype=counts.dtype)
        padded_counts[1:] = counts
        indptr = padded_counts.cumsum(dim=0)
        x = segment_csr(src=x, indptr=indptr, reduce=self.aggregation)

        # sanity check: dst_indices has len of batch_size * num_supernodes and has to be divisible by batch_size
        # if num_supernodes is not set in dataset this assertion fails
        if batch_idx is None:
            batch_size = 1
        else:
            batch_size = int(batch_idx.max().item()) + 1
            assert dst_indices.numel() % batch_size == 0

        return x, batch_size

    def forward(
        self,
        input_pos: torch.Tensor,
        supernode_idx: torch.Tensor,
        batch_idx: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass of the supernode pooling layer.

        Args:
            input_pos: Sparse tensor with shape (batch_size * number_of_points_per_sample, 3), representing the point cloud representation of the input geometry.
            supernode_idx: indexes of the supernodes in the sparse tensor input_pos.
            batch_idx: 1D tensor, containing the batch index of each entry in input_pos. Default None.
            input_features: Sparse tensor with shape (batch_size * number_of_points_per_sample, number_of_features)


        Returns:
            Tensor with the aggregated messages for each supernode.
        """
        assert input_pos.ndim == 2, f"input_pos has to be 2D, but has shape {input_pos.shape}"
        assert supernode_idx.ndim == 1, f"supernode_idx has to be 1D, but has shape {supernode_idx.shape}"
        assert self.input_features_dim is None or input_features is not None, (
            "input_features has to be set if num_input_features is set"
        )
        if input_features is not None and input_features.ndim == 3:
            input_features = einops.rearrange(
                input_features, "batch_size num_points features -> (batch_size num_points) features"
            )

        assert (
            input_features is None or input_features.ndim == 2 and input_features.shape[1] == self.input_features_dim
        ), "input_features must match num_input_features"

        src_idx, dst_idx = self.compute_src_and_dst_indices(
            batch_idx=batch_idx,
            input_pos=input_pos,
            supernode_idx=supernode_idx,
        )

        x, supernode_pos_embed = self.create_messages(
            input_pos=input_pos,
            src_idx=src_idx,
            dst_idx=dst_idx,
            supernode_idx=supernode_idx,
            input_features=input_features,
        )

        x, batch_size = self.accumulate_messages(x, dst_idx, supernode_idx, batch_idx)

        # convert to dense tensor (dim last)
        x = einops.rearrange(
            x,
            "(batch_size num_supernodes) dim -> batch_size num_supernodes dim",
            batch_size=batch_size,
        )

        if self.readd_supernode_pos and callable(self.proj):
            supernode_pos_embed = einops.rearrange(
                supernode_pos_embed,
                "(batch_size num_supernodes) dim -> batch_size num_supernodes dim",
                batch_size=len(x),
            )
            # concatenate input and supernode embeddings
            x = torch.concat([x, supernode_pos_embed], dim=-1)
            x = self.proj(x)

        return x
