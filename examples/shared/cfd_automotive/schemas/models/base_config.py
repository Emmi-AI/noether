#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from pydantic import BaseModel, Field

from noether.core.schemas.dataset import AeroDataSpecs


class TutorialBaseModelConfig(BaseModel):
    name: str = Field(...)
    """Name of the model, also used as identifier when saving/loading checkpoints and finding the correct model schema."""
    hidden_dim: int = Field(...)
    """Hidden dimension of the model."""
    kind: str = Field(...)
    """Kind of model to use, i.e. class path (tutorials.models.<model_class>)."""
    position_projection: Literal["linear", "sincos"] = "sincos"
    """String to indicate the type of position projection to use. Can be "sincos" or "linear". Defaults to "sincos"."""
    use_output_projection: bool = False
    """Boolean to indicate to use the output projection. Defaults to False."""
    use_bias_layers: bool = Field(True)
    """Boolean to indicate to use bias layers. Defaults to True."""
    data_specs: AeroDataSpecs
    """Data specifications for the model. If None, default data specifications will be used."""
