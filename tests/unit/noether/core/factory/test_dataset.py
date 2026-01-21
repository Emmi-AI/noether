#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from noether.core.factory.dataset import DatasetFactory


def test_init_defaults():
    """Test default initialization of wrapper factory."""
    with patch("noether.core.factory.dataset.Factory") as MockBaseFactory:
        factory = DatasetFactory()
        # Should create a default Factory for wrappers if none provided:
        assert factory.dataset_wrapper_factory == MockBaseFactory.return_value


def test_instantiate_no_wrappers():
    """Test instantiation when config has no wrappers."""
    factory = DatasetFactory()

    mock_config = SimpleNamespace(
        dataset_wrappers=None, kind="MyDataset", included_properties=None, excluded_properties=None
    )

    # Mock super().instantiate behavior
    # We patch 'noether.core.factory.dataset.Factory.instantiate' which is the parent method
    with patch("noether.core.factory.dataset.Factory.instantiate") as mock_super_inst:
        mock_dataset = SimpleNamespace(config=SimpleNamespace(included_properties=None, excluded_properties=None))
        mock_super_inst.return_value = mock_dataset

        result = factory.instantiate(mock_config)

        assert result == SimpleNamespace(config=SimpleNamespace(included_properties=None, excluded_properties=None))
        mock_super_inst.assert_called_once_with(mock_config)


def test_instantiate_with_wrappers():
    """Test that wrappers are applied recursively."""
    mock_wrapper_factory = MagicMock()
    factory = DatasetFactory(dataset_wrapper_factory=mock_wrapper_factory)

    wrapper1_cfg = SimpleNamespace(kind="Wrapper1")
    wrapper2_cfg = SimpleNamespace(kind="Wrapper2")
    mock_config = SimpleNamespace(
        dataset_wrappers=[wrapper1_cfg, wrapper2_cfg],
        kind="BaseDataset",
        included_properties=None,
        excluded_properties=None,
    )

    with patch("noether.core.factory.dataset.Factory.instantiate") as mock_super_inst:
        # Base creation:
        base_dataset = SimpleNamespace(config=SimpleNamespace(included_properties=None, excluded_properties=None))
        mock_super_inst.return_value = base_dataset

        # First call returns v1, second call returns v2:
        mock_wrapper_factory.instantiate.side_effect = ["dataset_v1", "dataset_v2"]

        result = factory.instantiate(mock_config)

        assert result == "dataset_v2"
        assert mock_wrapper_factory.instantiate.call_count == 2
