#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
from torch import nn

from noether.core.schemas.modules.layers import UnquantizedDropPathConfig


class UnquantizedDropPath(nn.Module):
    """Unquantized drop path (Stochastic Depth, https://arxiv.org/abs/1603.09382) per sample. Unquantized means
    that dropped paths are still calculated. Number of dropped paths is fully stochastic, i.e., it can happen that
    not a single path is dropped or that all paths are dropped. In a quantized drop path, the same amount of
    paths are dropped in each forward pass, resulting in large speedups with high drop_prob values. See
    https://arxiv.org/abs/2212.04884 for more discussion. UnquantizedDropPath does not provide any speedup,
    consider using a quantized version if large drop_prob values are used.

    Adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py#L150
    """

    def __init__(self, config: UnquantizedDropPathConfig) -> None:
        """
        Initialize the UnquantizedDropPath module.

        Args:
            config: Configuration for the UnquantizedDropPath module. See :class:`~noether.core.schemas.modules.layers.UnquantizedDropPathConfig` for the available options.
        """

        super().__init__()
        if not 0.0 <= config.drop_prob <= 1.0:
            raise ValueError("drop_prob must be in the range [0.0, 1.0]")
        self.drop_prob = config.drop_prob
        self.scale_by_keep = config.scale_by_keep

    @property
    def keep_prob(self):
        """Return the keep probability. I.e. the probability to keep a path, which is 1 - drop_prob.

        Returns:
            Float value of the keep probability.
        """

        return 1 - self.drop_prob

    def forward(self, x: torch.Tensor):
        """Forward function of the UnquantizedDropPath module.

        Args:
            x: Tensor to apply the drop path. Shape: (batch_size, ...).

        Returns:
            Tensor with drop path applied. Shape: (batch_size, ...). If drop_prob is 0, the input tensor is returned. If drop_prob is 1, a tensor with zeros is returned.
        """

        if self.drop_prob == 0.0 or not self.training:
            return x
        # work with arbitrary shape
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(self.keep_prob)
        if self.keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(self.keep_prob)
        return x * random_tensor

    def extra_repr(self):
        """Extra representation of the UnquantizedDropPath module.

        Returns:
            Return a string representation of the module.
        """
        return f"drop_prob={self.drop_prob:0.2f}"
