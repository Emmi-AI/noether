#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

from noether.core.optimizer.param_group_modifiers.base import ParamGroupModifierBase

if TYPE_CHECKING:
    import torch
    from torch import nn

    from noether.core.schemas.optimizers import CompletePModifierConfig


class CompletePModifier(ParamGroupModifierBase):
    """Applies CompleteP per-parameter scaling for learning rate, weight decay, and Adam epsilon.

    Classifies parameters into 5 groups based on name patterns and tensor dimensionality:
    1. Embedding params (outside blocks): lr_scale=1.0, eps=base_eps/m_w
    2. Hidden norm params (inside blocks, ndim<=1, not bias): lr_scale=depth_lr, wd=0, eps=hidden_eps
    3. Hidden weight params (inside blocks, ndim>=2): lr_scale=1/m_w * depth_lr, wd=base_wd*m_w, eps=hidden_eps
    4. Hidden bias params (inside blocks, bias): lr_scale=depth_lr, wd=0, eps=hidden_eps
    5. Final norm params (outside blocks, ndim<=1): lr_scale=1.0, wd=0, eps=base_eps/m_w

    When ``optimizer_type == "muon"`` (for :class:`~noether.core.optimizer.MuonComposite`):
    - ``eps`` is omitted from all groups. Muon's ``eps`` controls NS spectral regularization
      (different semantics from Adam), and the secondary optimizer (e.g. Lion) has no ``eps``.
    - The ``1/m_w`` width LR scaling on hidden weight matrices (group 3) is dropped because
      Muon's Newton-Schulz orthogonalization already bounds the update spectral norm
      independent of width. Hidden weights then receive only the depth scaling.
    - The ``m_w`` weight-decay scaling on hidden weights is also dropped, matching common
      Muon-CompleteP recipes (e.g. Moonlight) that use a single tuned weight decay for Muon.

    Reference: "Don't be lazy: CompleteP enables compute-efficient deep transformers" (NeurIPS 2025).
    """

    def __init__(self, completep_modifier_config: CompletePModifierConfig):
        super().__init__()
        cfg = completep_modifier_config
        self.optimizer_type = cfg.optimizer_type
        self.m_w = cfg.width_multiplier
        self.m_d = cfg.depth_multiplier
        self.depth_alpha_exp = cfg.depth_alpha_exp
        self.base_eps = cfg.base_eps
        self.base_weight_decay = cfg.base_weight_decay
        self.hidden_param_substrings = cfg.hidden_param_substrings

        # Pre-compute scaling factors
        self.depth_lr_scaling = self.m_d ** (self.depth_alpha_exp - 1)
        # Muon's NS orthogonalization bounds the update spectral norm independent of width,
        # so hidden weights skip both the 1/m_w LR scaling and the m_w WD scaling.
        if self.optimizer_type == "muon":
            self.hidden_weight_lr_scaling = self.depth_lr_scaling
            self.hidden_weight_wd = self.base_weight_decay
        else:
            self.hidden_weight_lr_scaling = (1.0 / self.m_w) * self.depth_lr_scaling
            self.hidden_weight_wd = self.base_weight_decay * self.m_w
        self.emb_unemb_eps = self.base_eps / self.m_w
        self.hidden_eps = self.base_eps / self.m_w * (self.m_d ** (-self.depth_alpha_exp))

        self._applied = False

    def _is_hidden_param(self, name: str) -> bool:
        return any(s in name for s in self.hidden_param_substrings)

    def _is_bias(self, name: str) -> bool:
        return name.split(".")[-1] == "bias"

    def get_properties(self, model: nn.Module, name: str, param: torch.Tensor) -> dict[str, float]:
        self._applied = True

        if self._is_hidden_param(name):
            # Inside transformer blocks
            if self._is_bias(name):
                # Hidden biases: no weight decay, depth-scaled LR
                props = dict(
                    lr_scale=self.depth_lr_scaling,
                    weight_decay=0.0,
                    eps=self.hidden_eps,
                )
            elif param.ndim <= 1:
                # Hidden norm params (LayerNorm weight, LayerScale gamma): no weight decay, depth-scaled LR
                props = dict(
                    lr_scale=self.depth_lr_scaling,
                    weight_decay=0.0,
                    eps=self.hidden_eps,
                )
            else:
                # Hidden weight matrices: width-scaled LR, width-scaled WD, depth-scaled LR
                # (Muon mode drops the width factors; see class docstring.)
                props = dict(
                    lr_scale=self.hidden_weight_lr_scaling,
                    weight_decay=self.hidden_weight_wd,
                    eps=self.hidden_eps,
                )
        else:
            # Outside transformer blocks (embedding, final norm, etc.)
            if param.ndim <= 1:
                # Final norm / non-block 1D params: no weight decay
                props = dict(
                    lr_scale=1.0,
                    weight_decay=0.0,
                    eps=self.emb_unemb_eps,
                )
            else:
                # Embedding weights / non-block 2D params
                props = dict(
                    lr_scale=1.0,
                    weight_decay=self.base_weight_decay,
                    eps=self.emb_unemb_eps,
                )

        if self.optimizer_type == "muon":
            props.pop("eps", None)
        return props

    def __str__(self):
        return (
            f"{type(self).__name__}("
            f"optimizer_type={self.optimizer_type}, "
            f"m_w={self.m_w}, m_d={self.m_d}, "
            f"alpha={self.depth_alpha_exp}, "
            f"base_eps={self.base_eps}, base_wd={self.base_weight_decay})"
        )

    def was_applied_successfully(self) -> bool:
        return self._applied
