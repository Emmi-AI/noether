#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

from noether.data.pipeline.sample_processor import SampleProcessor


class DuplicateKeysSampleProcessor(SampleProcessor):
    """Utility processor that simply duplicates the dictionary keys in a batch.

    Duplicates keys in the batch if they are in the key_map.
    Creates a new dictionary whose keys are duplicated but uses references to the values of the old dict.
    This avoids copying the data and at the same time does not modify this function's input.

    .. code-block:: python

        # dummy example
        processor = DuplicateKeysSampleProcessor(key_map={"original_key": "duplicated_key"})

        input_sample = {
            "original_key": tensor_data,
        }

        output_sample = processor(input_sample)
        # output_sample['original_key'] will be tensor_data
        # output_sample['duplicated_key'] will also be tensor_data

    """

    def __init__(self, key_map: dict[str, str]):
        """

        Args:
            key_map: Dict with source keys as keys and target keys as values. The source keys are duplicated
                in the samples and the target keys are created. The values of the source keys are used for the target
                keys.
        """
        self.key_map = key_map

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """

        Args:
            input_sample: Dictionary with the tensors of a single sample.

        Returns:
            Preprocessed copy of `input_sample` with the specified keys duplicated.
        """
        # copy to avoid changing method input
        output_sample = self.save_copy(input_sample)

        for source_key, target_key in self.key_map.items():
            output_sample[target_key] = self.save_copy(output_sample[source_key])
        return output_sample
