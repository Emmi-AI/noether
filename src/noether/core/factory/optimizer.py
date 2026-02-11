#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Callable
from functools import partial

from noether.core.factory.base import Factory
from noether.core.factory.utils import class_constructor_from_class_path
from noether.core.optimizer import OptimizerWrapper
from noether.core.schemas.optimizers import OptimizerConfig


class OptimizerFactory(Factory):
    """Factory for creating optimizers.

    Handles wrapping into :class:`~noether.core.optimizer.OptimizerWrapper` by creating the corresponding constructor
    for the underlying :class:`~torch.optim.Optimizer`. Objects are returned as partials, as creating the optimizer
    requires the model parameters from the instantiated model.
    """

    def __init__(self):
        super().__init__(returns_partials=True)

    def instantiate(  # type: ignore[override]
        self,
        optimizer_config: OptimizerConfig,
    ) -> Callable[..., OptimizerWrapper]:
        """Instantiates the optimizer based on the provided configuration.

        Args:
            optimizer_config: Configuration for the optimizer to create. This config contains both the
                :class:`~torch.optim.Optimizer` and the :class:`~noether.core.optimizer.OptimizerWrapper` configurations.
                See :class:`~noether.core.schemas.optimizers.OptimizerConfig` for available options.

        Returns:
            A callable that initializes the :class:`~noether.core.optimizer.OptimizerWrapper`.
        """

        # extract OptimizerWrapper kwargs (e.g. clip_grad_value or exclude_bias_from_wd)
        # these should not be passed to the torch optimizer but to the OptimizerWrapper afterwards
        # the optimizer and the optimizer wrapper are configured as one config, we need to split them for the partials

        if optimizer_config.kind is None:
            raise ValueError("OptimizerConfig.kind must be specified to create the torch optimizer.")
        torch_optimizer_kind = class_constructor_from_class_path(class_path=optimizer_config.kind)
        optimizer_config_dict = optimizer_config.model_dump(
            exclude={"kind"} | optimizer_config.return_optim_wrapper_args().keys()
        )  # filter out all the kwargs of the wrapper
        if len(optimizer_config_dict) > 0:
            torch_optim_constructor = partial(torch_optimizer_kind, **optimizer_config_dict)

        return partial(
            OptimizerWrapper,
            torch_optim_ctor=torch_optim_constructor,
            optim_wrapper_config=OptimizerConfig(
                **optimizer_config.return_optim_wrapper_args()
            ),  # reinstantiate with only the wrapper args
        )
