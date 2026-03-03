#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Union

from .ab_upt_config import ABUPTConfig
from .composite_transformer_config import CompositeTransformerConfig
from .transformer_config import TransformerConfig
from .transolver_config import TransolverConfig
from .transolver_plusplus_config import TransolverPlusPlusConfig
from .upt_config import UPTConfig

# TODO move this to the tutorial folder

AnyModelConfig = Union[
    TransformerConfig,
    TransolverConfig,
    UPTConfig,
    ABUPTConfig,
    TransolverPlusPlusConfig,
    CompositeTransformerConfig,
]
