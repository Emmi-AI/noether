#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from pydantic import Field, model_validator

from noether.core.schemas.callbacks import CallBackBaseConfig


class SurfaceVolumeEvaluationMetricsCallbackConfig(CallBackBaseConfig):
    name: Literal["SurfaceVolumeEvaluationMetricsCallback"] = "SurfaceVolumeEvaluationMetricsCallbacks"
    dataset_key: str = Field(...)
    """Key of the dataset to evaluate on"""
    forward_properties: list[str] = []
    """List of properties in the dataset to be forwarded during inference."""
    chunked_inference: bool = False
    "If True, perform inference in chunks over the full simulation geometry"
    chunk_properties: list[str] = []
    """List of properties in the dataset to be chunked use for chunked. Some properties don't need to be chunked."""
    batch_size: int = Field(1)
    """Batch size for evaluation. Currently only batch_size=1 is supported."""
    chunk_size: int | None = None
    """Size of each chunk when performing chunked inference. Usually equal to the number of surface/volume points during training"""
    sample_size_property: str | None = Field(None)
    """Property in the batch to determine the sample size (i.e., the size of either the surface or volume mesh) to know how many chunks to make"""

    @model_validator(mode="after")
    def validate_config(self) -> "SurfaceVolumeEvaluationMetricsCallbackConfig":
        if self.batch_size != 1:
            raise ValueError("SurfaceVolumeEvaluationMetricsCallback only supports batch_size=1")
        if self.chunked_inference:
            if self.chunk_size is None:
                raise ValueError("chunk_size must be specified when chunked_inference is True")
            if not self.forward_properties:
                raise ValueError("forward_properties must be specified when chunked_inference is True")
            if not self.chunk_properties:
                raise ValueError("chunk_properties must be specified when chunked_inference is True")
        return self
