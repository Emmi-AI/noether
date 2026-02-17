#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
from pathlib import Path

import pytest
import yaml

from noether.core.configs.hyperparameters import Hyperparameters
from noether.core.schemas.schema import ConfigSchema


class MockHyperparameters(ConfigSchema):
    """A mock Pydantic model for testing."""


@pytest.fixture
def mock_params() -> MockHyperparameters:
    """Provides a mock hyperparameter instance for tests."""
    return MockHyperparameters(
        output_path="/tmp",
        datasets=dict(),
        model=dict(name="abc", kind="xyz"),
        trainer=dict(kind="mock", effective_batch_size=32, callbacks=[]),
    )


class TestHyperparameters:
    """Tests for the Hyperparameters utility class."""

    def test_save_resolved(self, mock_params: MockHyperparameters, tmp_path: Path, caplog):
        """
        Tests that save_resolved correctly saves the model to a YAML file
        and logs the action.
        """
        out_file = tmp_path / "hyperparameters.yaml"
        expected_dump = mock_params.model_dump()

        with caplog.at_level(logging.INFO):
            Hyperparameters.save_resolved(mock_params, out_file, exclude_unset=False)

        assert out_file.is_file()
        with open(out_file) as f:
            content = yaml.safe_load(f)
            # we only serialise this field
            content.pop("config_schema_kind", None)  # Remove added field for comparison
        assert content == expected_dump

        assert f"Dumped resolved hyperparameters to {out_file}" in caplog.text
