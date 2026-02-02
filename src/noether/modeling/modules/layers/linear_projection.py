#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
from torch import nn

from noether.core.schemas.modules.layers import LinearProjectionConfig
from noether.modeling.functional.init import init_trunc_normal_zero_bias


class LinearProjection(nn.Module):
    """LinearProjection is a linear projection layer that can be used for 1D, 2D, and 3D data."""

    def __init__(
        self,
        config: LinearProjectionConfig,
    ) -> None:
        """
        Initialize the LinearProjection.

        Args:
            config: The configuration of the LinearProjection.

        Raises:
            NotImplementedError: raises not implemented error if the number of dimensions of the input domain is bigger than 4.
        """

        super().__init__()

        self.project: nn.Linear | nn.Conv1d | nn.Conv2d | nn.Conv3d | nn.Identity
        self.init_weights = config.init_weights

        if config.optional and config.input_dim == config.output_dim:
            self.project = nn.Identity()
        elif config.ndim is None:
            self.project = nn.Linear(config.input_dim, config.output_dim, bias=config.bias)
        elif config.ndim == 1:
            self.project = nn.Conv1d(config.input_dim, config.output_dim, kernel_size=1, bias=config.bias)
        elif config.ndim == 2:
            self.project = nn.Conv2d(config.input_dim, config.output_dim, kernel_size=1, bias=config.bias)
        elif config.ndim == 3:
            self.project = nn.Conv3d(config.input_dim, config.output_dim, kernel_size=1, bias=config.bias)
        else:
            raise NotImplementedError("""LinearProjection only supports ndim=None, 1, 2, or 3.""")

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the MLP with a specific initialization. Options are "torch" (i.e., default) or
            "truncnormal002".

        Raises:
            NotImplementedError: raised if the specified initialization is not implemented.
        """

        if self.init_weights == "torch":
            pass
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            init_trunc_normal_zero_bias(self.project)
        elif self.init_weights == "zeros":
            assert isinstance(self.project, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
            nn.init.zeros_(self.project.weight)
            if self.project.bias is not None:
                nn.init.zeros_(self.project.bias)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of the LinearProjection.

        Args:
            x: Input tensor to the LinearProjection.

        Returns:
            Output tensor from the LinearProjection.
        """

        return self.project(x)  # type: ignore[no-any-return]
