#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from .emmi_wing import EmmiWingPreset


class EmmiWingHFPreset(EmmiWingPreset):
    """Emmi-Wing preset using the HuggingFace 248-case subset.

    Identical pipeline, normalizers, and model config as :class:`EmmiWingPreset`, but uses the HF dataset class
    with its own splits (200/24/24).
    """

    dataset_kind = "noether.data.datasets.cfd.EmmiWingHFDataset"
