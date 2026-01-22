#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
import torch

from noether.modeling.models import AnchoredBranchedUPT as ABUPTBackbone
from tutorial.schemas.models import ABUPTConfig

from .base import BaseModel


class ABUPT(BaseModel):
    """Implementation of the AB-UPT model."""

    def __init__(
        self,
        model_config: ABUPTConfig,
        **kwargs,
    ):
        """Initialize the AB-UPT model.

        Args:
            model_config: The configuration for the AB-UPT model.
        """

        super().__init__(model_config=model_config, **kwargs)

        self.ab_upt = ABUPTBackbone(
            config=model_config,
        )

    def forward(
        # geometry
        self,
        geometry_position: torch.Tensor,
        geometry_supernode_idx: torch.Tensor,
        geometry_batch_idx: torch.Tensor | None,
        # anchors
        surface_anchor_position: torch.Tensor,
        volume_anchor_position: torch.Tensor,
        # design parameters
        geometry_design_parameters: torch.Tensor | None = None,
        inflow_design_parameters: torch.Tensor | None = None,
        # queries
        query_surface_position: torch.Tensor | None = None,
        query_volume_position: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the AB-UPT model.

        Args:
            geometry_position: Positions of the geometry points.
            geometry_supernode_idx: Indices of the supernodes for the geometry points.
            geometry_batch_idx: Batch indices for the geometry points.
            surface_position: Positions of the surface anchor points.
            volume_position: Positions of the volume anchor points.
            geometry_design_parameters: Design parameters for the geometry.
            inflow_design_parameters: Design parameters for the inflow.
            query_surface_position: Query positions for the surface points.
            query_volume_position: Query positions for the volume points.

        Returns:
            A dictionary containing the model outputs.
        """

        return self.ab_upt(
            # geometry
            geometry_position=geometry_position,
            geometry_supernode_idx=geometry_supernode_idx,
            geometry_batch_idx=geometry_batch_idx,
            # anchors
            surface_anchor_position=surface_anchor_position,
            volume_anchor_position=volume_anchor_position,
            # design parameters
            geometry_design_parameters=geometry_design_parameters,
            inflow_design_parameters=inflow_design_parameters,
            # queries
            query_surface_position=query_surface_position,
            query_volume_position=query_volume_position,
        )
