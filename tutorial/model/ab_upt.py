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
        """Forward pass of the AB-UPT model."""

        # Build domain dicts from the legacy surface/volume args
        domain_anchor_positions = {
            "surface": surface_anchor_position,
            "volume": volume_anchor_position,
        }
        domain_query_positions: dict[str, torch.Tensor] = {}
        if query_surface_position is not None:
            domain_query_positions["surface"] = query_surface_position
        if query_volume_position is not None:
            domain_query_positions["volume"] = query_volume_position

        conditioning_inputs: dict[str, torch.Tensor] = {}
        if geometry_design_parameters is not None:
            conditioning_inputs["geometry_design_parameters"] = geometry_design_parameters
        if inflow_design_parameters is not None:
            conditioning_inputs["inflow_design_parameters"] = inflow_design_parameters

        out, _ = self.ab_upt(
            geometry_position=geometry_position,
            geometry_supernode_idx=geometry_supernode_idx,
            geometry_batch_idx=geometry_batch_idx,
            domain_anchor_positions=domain_anchor_positions,
            domain_query_positions=domain_query_positions or None,
            conditioning_inputs=conditioning_inputs or None,
        )
        return out
