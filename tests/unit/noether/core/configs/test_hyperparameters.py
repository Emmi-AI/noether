#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import ClassVar

import pytest
import yaml
from pydantic import BaseModel, RootModel

from noether.core.configs.hyperparameters import Hyperparameters, _inject_discriminator_fields
from noether.core.schemas.lib import _RegistryBase
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
        trainer=dict(kind="mock", effective_batch_size=32, callbacks=[], max_epochs=1),
        spec=DimSpec({"def": 1, "abc": 2}),
    )


class _RegistryStub(_RegistryBase):
    """Minimal _RegistryBase concrete subclass for unit tests."""

    _type_field: ClassVar[str] = "kind"
    _registry: ClassVar[dict] = {}

    kind: str | None = "registry_stub.default"
    value: int = 0


class _NestedModel(BaseModel):
    """Wraps a registry config + a list/dict of registry configs to exercise recursion."""

    pipeline: _RegistryStub
    callbacks: list[_RegistryStub] = []
    by_key: dict[str, _RegistryStub] = {}


class TestInjectDiscriminatorFields:
    """`_inject_discriminator_fields` re-adds `kind` wherever
    `model_dump(exclude_unset=True)` dropped it."""

    def test_injects_kind_when_unset(self):
        model = _RegistryStub(value=42)  # `kind` not explicitly passed
        dumped = model.model_dump(exclude_unset=True)

        assert "kind" not in dumped  # baseline: dump strips the discriminator

        _inject_discriminator_fields(model, dumped)

        assert dumped["kind"] == "registry_stub.default"

    def test_preserves_user_set_kind(self):
        model = _RegistryStub(kind="custom_kind", value=42)
        dumped = model.model_dump(exclude_unset=True)

        _inject_discriminator_fields(model, dumped)

        assert dumped["kind"] == "custom_kind"

    def test_recurses_into_nested_models(self):
        model = _NestedModel(pipeline=_RegistryStub(value=1))
        dumped = model.model_dump(exclude_unset=True)

        assert "kind" not in dumped["pipeline"]

        _inject_discriminator_fields(model, dumped)

        assert dumped["pipeline"]["kind"] == "registry_stub.default"

    def test_recurses_into_lists(self):
        model = _NestedModel(
            pipeline=_RegistryStub(value=1),
            callbacks=[_RegistryStub(value=2), _RegistryStub(kind="explicit", value=3)],
        )
        dumped = model.model_dump(exclude_unset=True)

        _inject_discriminator_fields(model, dumped)

        assert dumped["callbacks"][0]["kind"] == "registry_stub.default"
        assert dumped["callbacks"][1]["kind"] == "explicit"

    def test_recurses_into_dicts(self):
        model = _NestedModel(
            pipeline=_RegistryStub(value=1),
            by_key={"a": _RegistryStub(value=2), "b": _RegistryStub(kind="b_kind", value=3)},
        )
        dumped = model.model_dump(exclude_unset=True)

        _inject_discriminator_fields(model, dumped)

        assert dumped["by_key"]["a"]["kind"] == "registry_stub.default"
        assert dumped["by_key"]["b"]["kind"] == "b_kind"


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
