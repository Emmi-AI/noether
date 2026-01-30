#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence
from typing import Any

import torch

from noether.data.pipeline.sample_processor import SampleProcessor
from noether.modeling.functional.logscale import from_logscale, to_logscale


class MomentNormalizationSampleProcessor(SampleProcessor):
    """Normalizes a value with its mean and standard deviation (i.e., its moments).


    .. code-block:: python
        # dummy example
        processor = MomentNormalizationSampleProcessor(
            item="measurement",
            mean=[10.0],
            std=[2.0],
            logscale=False,
        )
        input_sample = {
            "measurement": torch.tensor([[12.0], [14.0], [8.0]]),
            "other_item": torch.tensor([[1.0], [2.0], [3.0]]),
        }
        output_sample = processor(input_sample)
        # output_sample['measurement'] will be tensor([[1.0], [2.0], [-1.0]])
        # output_sample['other_item'] will be unchanged.
    """

    def __init__(
        self,
        item: str,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        logmean: Sequence[float] | None = None,
        logstd: Sequence[float] | None = None,
        logscale: bool = False,
    ):
        """

        Args:
            item: The item (i.e., key in the input sample dictionary) to normalize.
            mean: The mean of the value. Mandatory if logscale=False.
            std: The standard deviation of the value. Mandatory if logscale=False.
            logmean: The mean of the value in logscale. Mandatory if logscale=True.
            logstd: The standard deviation of the value in logscale. Mandatory if logscale=True.
            logscale: Whether to convert the value to logscale before normalization.
        """
        if logscale:
            if logmean is None or logstd is None:
                raise ValueError("logmean and logstd must be set if logscale=True")
            if len(logmean) != len(logstd):
                raise RuntimeError("Mean and standard deviation must have the same length.")
        else:
            if mean is None or std is None:
                raise ValueError("mean and standard deviation must be set if logscale=False")
            if len(mean) != len(std):
                raise RuntimeError("Mean and standard deviation must have the same length.")

        self.item = item
        self.mean_tensor = None if mean is None else torch.tensor(mean).unsqueeze(0)
        self.std_tensor = None if std is None else torch.tensor(std).unsqueeze(0)
        self.logmean_tensor = None if logmean is None else torch.tensor(logmean).unsqueeze(0)
        self.logstd_tensor = None if logstd is None else torch.tensor(logstd).unsqueeze(0)
        self.logscale = logscale

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """Pre-processes data on a sample-level to normalize a value to approximately mean=0 std=1.

        Args:
            input_sample: Dictionary with the tensors of a single sample.

        Return:
           Preprocessed copy of `input_sample` with the specified item normalized.
        """
        # copy to avoid changing method input
        output_sample = self.save_copy(input_sample)
        # a dict will be shallow-copied, thus cloning is needed:
        x = output_sample[self.item].clone()

        if self.logscale:
            x = to_logscale(x)
            x.sub_(self.logmean_tensor).div_(self.logstd_tensor)  # type: ignore[arg-type]
        else:
            x.sub_(self.mean_tensor).div_(self.std_tensor)  # type: ignore[arg-type]

        output_sample[self.item] = x
        return output_sample

    def inverse(self, key: str, value: torch.Tensor) -> tuple[str, torch.Tensor]:
        """Inverts the normalization from the __call__ method of a single item in the batch.

        Args:
            key: The name of the item.
            value: The value of the item.

        Returns:
            (key, value): The same name and the denormalized value.
        """
        if key != self.item:
            return key, value
        if self.logscale:
            assert self.logmean_tensor is not None and self.logstd_tensor is not None
            if value.ndim == self.logmean_tensor.ndim:
                # sparse tensor -> no additional dimension needed
                denormalized_value = value * self.logstd_tensor.to(value.device) + self.logmean_tensor.to(value.device)
            else:
                # dense tensor -> add batch dimension
                logstd_tensor = self.logstd_tensor.unsqueeze(0).to(value.device)
                logmean_tensor = self.logmean_tensor.unsqueeze(0).to(value.device)
                denormalized_value = value * logstd_tensor + logmean_tensor
            denormalized_value = from_logscale(denormalized_value)
        else:
            assert self.mean_tensor is not None and self.std_tensor is not None
            if value.ndim == self.mean_tensor.ndim:
                # sparse tensor -> no additional dimension needed
                denormalized_value = value * self.std_tensor.to(value.device) + self.mean_tensor.to(value.device)
            else:
                # dense tensor -> add batch dimension
                std_tensor = self.std_tensor.unsqueeze(0).to(value.device)
                mean_tensor = self.mean_tensor.unsqueeze(0).to(value.device)
                denormalized_value = value * std_tensor + mean_tensor

        return key, denormalized_value
