#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

from noether.core.optimizer.param_group_modifiers.base import ParamGroupModifierBase

if TYPE_CHECKING:
    import torch
    from torch import nn

    from noether.core.optimizer.schemas import ParamGroupModifierConfig


class WeightDecayByNameModifier(ParamGroupModifierBase):
    """Changes the weight decay value for a single parameter. Use-cases:
    - ViT exclude CLS token parameters
    - Transformer learned positional embeddings
    - Learnable query tokens for cross attention ("PerceiverPooling")
    """

    def __init__(self, param_group_modifier_config: ParamGroupModifierConfig):
        super().__init__()

        self.name = param_group_modifier_config.name
        self.value = param_group_modifier_config.value
        self.param_was_found = False

    def get_properties(self, model: nn.Module, name: str, param: torch.Tensor) -> dict[str, float]:
        """This method is called with all items of `model.named_parameters()` to compose the parameter groups for the
        whole model. If the desired parameter name is found, it returns a modifier that sets the weight decay.

        Args:
            model: Model from which the parameter originates from. Used to extract properties (e.g., number of layers
                for a layerwise learning rate decay).
            name: Name of the parameter as stored inside the model.
            param: The parameter tensor.
        """
        if name == self.name:
            if self.param_was_found:
                raise RuntimeError(f"found two parameters matching name '{self.name}'")

            self.param_was_found = True
            return dict(weight_decay=self.value or 0.0)
        return {}

    def __str__(self):
        return f"{type(self).__name__}(name={self.name}), value={self.value})"

    def was_applied_successfully(self):
        """Check if the parameter was found within the model."""
        return self.param_was_found
