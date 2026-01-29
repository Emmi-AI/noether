#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import numpy as np
import torch

from noether.core.callbacks import PeriodicCallback
from noether.core.extractors import ForwardHook
from noether.core.utils.common import select_with_path


class SupernodePoolingCoverageCallback(PeriodicCallback):
    """Hooks into the message MLP of the supernode pooling to retrieve the coverage/degree of the radius graph."""

    def __init__(self, supernode_pooling_path="encoder", **kwargs):
        super().__init__(**kwargs)
        self.supernode_pooling_path = supernode_pooling_path
        self.hook_outputs: dict[str, torch.Tensor] = {}
        self.pos_embed_hook = ForwardHook(outputs=self.hook_outputs, key="pos_embed")
        self.dst_idx_hook = ForwardHook(outputs=self.hook_outputs, key="dst_idx")
        self.src_idx_hook = ForwardHook(outputs=self.hook_outputs, key="src_idx")
        self.tracked_degrees: list[float] = []
        self.tracked_coverages: list[float] = []

    def before_training(self, **_) -> None:
        encoder = select_with_path(self.model, path=self.supernode_pooling_path)
        encoder.pos_embed.register_forward_hook(self.pos_embed_hook)  # type: ignore[attr-defined]
        encoder.dst_idx_hook.register_forward_hook(self.dst_idx_hook)  # type: ignore[attr-defined]
        encoder.src_idx_hook.register_forward_hook(self.src_idx_hook)  # type: ignore[attr-defined]

    def track_after_accumulation_step(self, *_, **__) -> None:
        assert self.hook_outputs.keys() == {"pos_embed", "dst_idx", "src_idx"}
        pos_embed = self.hook_outputs["pos_embed"]
        src_idx = self.hook_outputs["src_idx"]
        dst_idx = self.hook_outputs["dst_idx"]
        assert torch.is_tensor(pos_embed) and pos_embed.ndim == 2
        assert torch.is_tensor(src_idx) and src_idx.ndim == 1
        assert torch.is_tensor(dst_idx) and dst_idx.ndim == 1
        num_supernodes = dst_idx.unique().numel()
        num_edges = len(dst_idx)
        num_inputs = len(pos_embed)
        degree = num_edges / num_supernodes
        coverage = src_idx.unique().numel() / num_inputs
        self.tracked_degrees.append(degree)
        self.tracked_coverages.append(coverage)
        self.hook_outputs.clear()

    def periodic_callback(self, **_) -> None:
        degree = float(np.mean(self.tracked_degrees))
        coverage = float(np.mean(self.tracked_coverages))
        self.writer.add_scalar(
            key=f"spool/degree/{self.to_short_interval_string()}",
            value=degree,
            logger=self.logger,
            format_str=".2f",
        )
        self.writer.add_scalar(
            key=f"spool/coverage/{self.to_short_interval_string()}",
            value=coverage,
            logger=self.logger,
            format_str=".2f",
        )
        self.tracked_degrees.clear()
        self.tracked_coverages.clear()
