#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Union

from .property_subset import PropertySubsetWrapper
from .repeat import RepeatWrapper, RepeatWrapperConfig
from .shuffle import ShuffleWrapper, ShuffleWrapperConfig
from .subset import SubsetWrapper, SubsetWrapperConfig
from .timing import META_GETITEM_TIME, TimingWrapper

DatasetWrappers = Union[RepeatWrapperConfig, ShuffleWrapperConfig, SubsetWrapperConfig]

__all__ = [
    "DatasetWrappers",
    "META_GETITEM_TIME",
    "PropertySubsetWrapper",
    "RepeatWrapper",
    "RepeatWrapperConfig",
    "ShuffleWrapper",
    "ShuffleWrapperConfig",
    "SubsetWrapper",
    "SubsetWrapperConfig",
    "TimingWrapper",
]
