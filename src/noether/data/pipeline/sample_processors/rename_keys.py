#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

from noether.data.pipeline.sample_processor import SampleProcessor


class RenameKeysSampleProcessor(SampleProcessor):
    """Sample processor that simply renames the dictionary keys in a batch.
    Rename keys in the batch if they are in the key_map and keep old keys otherwise.
    Creates a new dictionary whose keys are renamed but uses references to the values of the old dict.
    This avoids copying the data and at the same time does not modify this function's input.

    .. code-block:: python
        # dummy example
        processor = RenameKeysSampleProcessor(key_map={"old_key1": "new_key1", "old_key2": "new_key2"})
        input_sample = {
            "old_key1": some_tensor1,
            "old_key2": some_tensor2,
            "unchanged_key": some_tensor3,
        }

        output_sample = processor(input_sample)
        # output_sample will be: {
        #     'new_key1': some_tensor1,
        #     'new_key2': some_tensor2,
        #     'unchanged_key': some_tensor3,
        # }

    """

    def __init__(self, key_map: dict[str, str]):
        """

        Args:
            key_map: Dict with source keys as keys and target keys as values. The source keys are renamed target keys.
        """

        self.key_map = key_map

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """

        Args:
            input_sample: Dictionary with the tensors of a single sample.

        Returns:
            Preprocessed copy of `input_sample` with the keys renamed according to the key_map.
        """
        # copy to avoid changing method input
        output_sample = self.save_copy(input_sample)
        return {self.key_map.get(key, key): value for key, value in output_sample.items()}
