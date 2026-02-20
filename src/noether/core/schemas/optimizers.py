#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Self

from pydantic import BaseModel, Field, model_validator

from noether.core.schemas.schedules import AnyScheduleConfig


class ParamGroupModifierConfig(BaseModel):
    """Configuration for a parameter group modifier. Both for the LrScaleByNameModifier and the WeightDecayByNameModifier,"""

    kind: str | None = None
    """The class path of the parameter group modifier. Either noether.core.optimizer.param_group_modifiers.LrScaleByNameModifier or noether.core.optimizer.param_group_modifiers.WeightDecayByNameModifier."""
    scale: float | None = Field(None, ge=0.0)
    """The scaling factor for the learning rate. Must be greater than 0.0. Only for the LrScaleByNameModifier."""
    value: float | None = Field(None, ge=0.0)
    """The weight decay value. With 0.0  the parameter is excluded from the weight decay. Only for the WeightDecayByNameModifier."""
    name: str
    """The name of the parameter within the model. E.g., 'backbone.cls_token'."""

    @model_validator(mode="after")
    def check_scale_or_value_exclusive(self) -> Self:
        """
        Validates that either 'scale' or 'value' is provided, but not both.
        This is a model-level validator that runs after individual field validation.
        """
        # Case 1: Both are provided (which is invalid)
        if self.scale is not None and self.value is not None:
            raise ValueError("Provide either 'scale' or 'value', but not both.")

        # Case 2: Neither is provided (which is also invalid)
        if self.scale is None and self.value is None:
            raise ValueError("Either 'scale' or 'value' must be provided.")

        # If one of the above conditions isn't met, the data is valid.
        return self


class OptimizerConfig(BaseModel):
    model_config = {"extra": "forbid"}

    kind: str | None = None
    """The class path of the torch optimizer to use. E.g., 'torch.optim.AdamW'."""
    lr: float | None = Field(None, gt=0.0)
    """The learning rate for the optimizer."""
    weight_decay: float | None = Field(0.0, ge=0.0)
    """The weight decay (L2 penalty) for the optimizer."""

    # these are the kwargs for the OptimWrapper
    clip_grad_value: float | None = Field(None, ge=0.0)
    """The maximum value for gradient clipping."""
    clip_grad_norm: float | None = Field(None, ge=0.0)
    """The maximum norm for gradient clipping."""
    param_group_modifiers_config: list[ParamGroupModifierConfig] | None = None
    """List of parameter group modifiers to apply. These can modify the learning rate or weight decay for specific parameters."""
    exclude_bias_from_weight_decay: bool = True
    """If true, excludes the bias parameters (i.e., parameters that end with '.bias') from the weight decay. Default true."""
    exclude_normalization_params_from_weight_decay: bool = True
    """If true, excludes the weights of normalization layers from the weight decay. This is implemented by excluding all 1D tensors from the weight decay. Default true."""
    weight_decay_schedule: AnyScheduleConfig | None = Field(None, discriminator="kind")
    schedule_config: AnyScheduleConfig | None = Field(None, discriminator="kind")

    _optim_wrapper_kwargs: set[str] = {
        "clip_grad_value",
        "clip_grad_norm",
        "param_group_modifiers_config",
        "exclude_bias_from_weight_decay",
        "exclude_normalization_params_from_weight_decay",
        "weight_decay_schedule",
        "schedule_config",
    }

    def return_optim_wrapper_args(self) -> dict:
        return self.model_dump(include=self._optim_wrapper_kwargs)
