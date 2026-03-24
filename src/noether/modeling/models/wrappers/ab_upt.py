#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch

from noether.core.models import Model
from noether.core.schemas.models import AnchorBranchedUPTConfig
from noether.modeling.models import AnchoredBranchedUPT


class ABUPTWrapper(Model):
    """Factory-compatible wrapper for the AnchoredBranchedUPT backbone.

    Bridges the factory's ``(config, **kwargs)`` instantiation pattern to the core model which only accepts
    a ``config`` argument.
    """

    def __init__(self, model_config: AnchorBranchedUPTConfig, **kwargs) -> None:
        super().__init__(model_config=model_config, **kwargs)
        self.backbone = AnchoredBranchedUPT(config=model_config)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        return self.backbone(**kwargs)  # type: ignore[no-any-return]
