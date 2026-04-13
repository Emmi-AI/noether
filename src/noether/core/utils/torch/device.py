#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any


def move_items_to_device(device, batch: dict[str, Any]):
    """Moves everything in the batch to the given device."""
    device_batch = {}
    for key in batch.keys():
        if isinstance(batch[key], list):
            assert len(batch[key]) == 1
            item = batch[key][0]
            device_batch[key] = None if item is None else item.to(device, non_blocking=True)
        else:
            device_batch[key] = None if batch[key] is None else batch[key].to(device, non_blocking=True)
    return device_batch
