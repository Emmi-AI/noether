#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import os
from collections import OrderedDict
from pathlib import Path

import pytest
import yaml
from pydantic import RootModel

from noether.core.configs.hyperparameters import Hyperparameters
from noether.core.schemas.schema import ConfigSchema


class DimSpec(RootModel[OrderedDict[str, int]]):
    pass


class MockHyperparameters(ConfigSchema):
    """A mock Pydantic model for testing."""

    spec: DimSpec


@pytest.fixture
def mock_params() -> MockHyperparameters:
    """Provides a mock hyperparameter instance for tests."""
    os.environ["MASTER_PORT"] = "12345"  # Set a fixed master port for testing
    return MockHyperparameters(
        output_path="/tmp",
        datasets=dict(),
        model=dict(name="abc", kind="xyz"),
        trainer=dict(kind="mock", effective_batch_size=32, callbacks=[]),
        spec=DimSpec({"def": 1, "abc": 2}),
    )


class TestHyperparameters:
    """Tests for the Hyperparameters utility class."""

    def test_save_resolved(self, mock_params: MockHyperparameters, tmp_path: Path, caplog):
        """
        Tests that save_resolved correctly saves the model to a YAML file
        and logs the action.
        """
        out_file = tmp_path / "hyperparameters.yaml"

        with caplog.at_level(logging.INFO):
            Hyperparameters.save_resolved(mock_params, out_file)

        assert out_file.is_file()
        with open(out_file) as f:
            content = yaml.safe_load(f)
            # we only serialise this field
            content.pop("config_schema_kind", None)  # Remove added field for comparison
        assert mock_params.model_dump(exclude_unset=True) == content

        loaded = MockHyperparameters.model_validate(content)
        assert mock_params == loaded

        assert f"Dumped resolved hyperparameters to {out_file}" in caplog.text
