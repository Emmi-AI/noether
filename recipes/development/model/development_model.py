#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import torch

from noether.core.models import Model
from noether.core.models.base import ModelBaseConfig


class DevelopmentModelConfig(ModelBaseConfig):
    kind: str = "development.DevelopmentModel"
    name: str = "development_model"
    input_dim: int
    hidden_dim: int = 256
    output_dim: int


class DevelopmentModel(Model):
    def __init__(self, model_config: DevelopmentModelConfig, **kwargs):
        super().__init__(model_config=model_config, **kwargs)

        self.layer1 = torch.nn.Linear(self.model_config.input_dim, self.model_config.hidden_dim)
        self.layer2 = torch.nn.Linear(self.model_config.hidden_dim, self.model_config.output_dim)

    def forward(self, x_z: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x_z))
        x = self.layer2(x)

        return {"output": x}
