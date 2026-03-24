#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch

from noether.core.models import Model
from noether.core.schemas.models import UPTConfig
from noether.modeling.models import UPT


class UPTWrapper(Model):
    """Factory-compatible wrapper for the UPT backbone.

    Combines separate surface/volume query positions into the single ``query_position`` that the core UPT expects,
    and splits the raw output tensor back into a named dict using ``data_specs``.
    """

    def __init__(self, model_config: UPTConfig, **kwargs):
        super().__init__(model_config=model_config, **kwargs)
        self.backbone = UPT(config=model_config)
        self.data_specs = model_config.data_specs

    def forward(
        self,
        surface_position_batch_idx: torch.Tensor,
        surface_position_supernode_idx: torch.Tensor,
        surface_position: torch.Tensor,
        surface_query_position: torch.Tensor,
        volume_query_position: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query_position = torch.cat([surface_query_position, volume_query_position], dim=1)

        x = self.backbone(
            surface_position_batch_idx=surface_position_batch_idx,
            surface_position_supernode_idx=surface_position_supernode_idx,
            surface_position=surface_position,
            query_position=query_position,
        )

        num_surface = surface_query_position.shape[1]
        return self._gather_outputs(x, num_surface)

    def _gather_outputs(self, x: torch.Tensor, num_surface: int) -> dict[str, torch.Tensor]:
        """Split the raw prediction tensor into named surface/volume outputs."""
        result: dict[str, torch.Tensor] = {}
        surface_out = x[:, :num_surface, :]
        volume_out = x[:, num_surface:, :]

        offset = 0
        for name, dim in self.data_specs.surface_output_dims:
            result[f"surface_{name}"] = surface_out[..., offset : offset + dim]
            offset += dim

        if self.data_specs.volume_output_dims is not None:
            offset = 0
            for name, dim in self.data_specs.volume_output_dims:
                result[f"volume_{name}"] = volume_out[..., offset : offset + dim]
                offset += dim

        return result
