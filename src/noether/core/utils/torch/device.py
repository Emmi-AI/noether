#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any


def move_items_to_device(device, batch: dict[str, Any]):
    """Moves everything in the batch to the given device."""
    # non_blocking=True is only safe with CUDA (pin_memory + streams).
    # MPS has no independent streams so non_blocking can retain source tensors.
    device_type = device.type if hasattr(device, "type") else str(device)
    non_blocking = device_type == "cuda"
    device_batch = {}
    for key in batch.keys():
        if isinstance(batch[key], list):
            assert len(batch[key]) == 1
            item = batch[key][0]
            device_batch[key] = None if item is None else item.to(device, non_blocking=non_blocking)
        else:
            device_batch[key] = None if batch[key] is None else batch[key].to(device, non_blocking=non_blocking)
    return device_batch
