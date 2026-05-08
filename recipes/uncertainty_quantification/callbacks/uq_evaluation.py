#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch
from aero_cfd.callbacks import AeroMetricsCallback, AeroMetricsCallbackConfig


class UQSurfaceVolumeEvaluationMetricsCallback(AeroMetricsCallback):
    """Evaluation callback for UQ models that output {field}_mean instead of {field}.

    Remaps model output keys so the parent class's denormalization and metric
    computation work unchanged.
    """

    def __init__(self, callback_config: AeroMetricsCallbackConfig, **kwargs):
        super().__init__(callback_config, **kwargs)

    def _run_model_inference(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = super()._run_model_inference(batch)

        # Remap {field}_mean -> {field} so parent class can find them
        remapped: dict[str, torch.Tensor] = {}
        for key, value in outputs.items():
            if key.endswith("_mean"):
                remapped[key[: -len("_mean")]] = value
            elif not key.endswith("_log_var"):
                remapped[key] = value
        return remapped
