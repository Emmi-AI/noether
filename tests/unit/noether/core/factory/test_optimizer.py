#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from functools import partial
from unittest.mock import MagicMock, patch

from noether.core.factory.optimizer import OptimizerFactory


def test_init_sets_returns_partials():
    """OptimizerFactory must always set returns_partials=True."""
    factory = OptimizerFactory()
    assert factory.returns_partials is True


def test_instantiate_logic():
    """
    Test the complex partial application logic:
    1. Splits config into torch args and wrapper args.
    2. Creates a partial for the torch optimizer.
    3. Returns a partial for OptimizerWrapper containing the torch partial.
    """
    factory = OptimizerFactory()

    # Mock the Configuration Object, it needs: .kind, .model_dump(), .return_optim_wrapper_args():
    mock_config = MagicMock()
    mock_config.kind = "torch.optim.Adam"

    # Setup return values
    wrapper_args = {"clip_grad": 1.0}
    mock_config.return_optim_wrapper_args.return_value = wrapper_args

    # Full dump (simulating what model_dump returns) logic says: model_dump(exclude={"kind"} | wrapper_keys)
    # We mock the result of that call directly.
    torch_args = {"lr": 0.001}
    mock_config.model_dump.return_value = torch_args

    with (
        patch("noether.core.factory.optimizer.class_constructor_from_class_path") as mock_class_loader,
        patch("noether.core.factory.optimizer.OptimizerWrapper") as MockWrapper,
        patch("noether.core.factory.optimizer.OptimizerConfig") as MockConfigConstructor,
    ):
        # Mock the torch optimizer class (e.g. the actual Adam class)
        MockTorchOptimizerClass = MagicMock()
        mock_class_loader.return_value = MockTorchOptimizerClass

        result_partial = factory.instantiate(mock_config)

        # Verify class loader was called with correct kind:
        mock_class_loader.assert_called_once_with(class_path="torch.optim.Adam")

        # Verify result is a partial of OptimizerWrapper:
        assert isinstance(result_partial, partial)
        assert result_partial.func == MockWrapper

        # Verify the partial contains the re-instantiated wrapper config.
        # The code calls OptimizerConfig(**wrapper_args):
        MockConfigConstructor.assert_called_with(**wrapper_args)
        assert result_partial.keywords["optim_wrapper_config"] == MockConfigConstructor.return_value

        # Verify the partial contains the torch optimizer constructor (which is also a partial):
        torch_optim_constructor = result_partial.keywords["torch_optim_ctor"]  # FIXME: the key is from the source code
        assert isinstance(torch_optim_constructor, partial)
        assert torch_optim_constructor.func == MockTorchOptimizerClass
        assert torch_optim_constructor.keywords == torch_args


def test_instantiate_no_torch_args():
    """Test edge case where torch optimizer has no extra args."""
    factory = OptimizerFactory()
    mock_config = MagicMock()
    mock_config.kind = "SimpleOptimizer"
    mock_config.return_optim_wrapper_args.return_value = dict()
    mock_config.model_dump.return_value = dict()  # No args for optimizer

    with patch("noether.core.factory.optimizer.class_constructor_from_class_path") as mock_loader:
        factory.instantiate(mock_config)
