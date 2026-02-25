#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import torch
import torch.nn as nnq

from noether.core.models import Model
from noether.core.schemas.models import ModelBaseConfig


class ModelConfig(ModelBaseConfig):
    # custom model config options can be added here
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 10


class Model(Model):
    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config=model_config, **kwargs)
        # define your model architecture here
        self.layer1 = nnq.Linear(model_config.input_dim, model_config.hidden_dim)
        self.layer2 = nnq.Linear(model_config.hidden_dim, model_config.output_dim)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Implement the forward pass of your model here")
