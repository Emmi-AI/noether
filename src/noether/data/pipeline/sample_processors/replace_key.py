#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

from noether.data.pipeline.sample_processor import SampleProcessor


class ReplaceKeySampleProcessor(SampleProcessor):
    """Sample processor that replaces the key with multiple other keys.
    Replaces a key in the batch with one or multiple other keys.
    Creates a new dictionary whose keys are duplicated but uses references to the values of the old dict.
    This avoids copying the data and at the same time does not modify this function's input.

    .. code-block:: python
        # dummy example
        processor = ReplaceKeySampleProcessor(source_key="source", target_keys={"target1", "target2"})
        input_sample = {
            "source": some_tensor,
            "unchanged_key": some_other_tensor,
        }
        output_sample = processor(input_sample)
        # output_sample will be: {
        #     'target1': some_tensor,
        #     'target2': some_tensor,
        #     'unchanged_key': some_other_tensor,
        # }
    """

    def __init__(self, source_key: str, target_keys: set[str]):
        """

        Args:
            source_key: Key in the input_sample to be replaced.
            target_keys: List of keys where source_key should be replaced in.
        """
        self.source_key = source_key
        self.target_keys = target_keys

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """

        Args:
            input_sample: Dictionary with the tensors of a single sample.

        Returns:
            Preprocessed copy of `input_sample` with the source key replaced by the target keys.
        """

        output_sample = self.save_copy(input_sample)
        source_item = output_sample.pop(self.source_key)
        for target_key in self.target_keys:
            output_sample[target_key] = self.save_copy(source_item)
        return output_sample
