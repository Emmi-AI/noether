#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Union

import pytest
import yaml
from pydantic import BaseModel, Field, RootModel

from noether.core.configs.hyperparameters import Hyperparameters, _inject_discriminator_fields
from noether.core.schemas.lib import Discriminated, _RegistryBase
from noether.core.schemas.schema import ConfigSchema
from noether.data.base.dataset import DatasetBaseConfig
from noether.data.preprocessors.normalizers import MeanStdNormalizerConfig


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


class _NormalizedDatasetConfig(DatasetBaseConfig):
    """Dataset config stub whose ``kind`` resolves back to this class on reload."""

    kind: str | None = "tests.unit.noether.core.configs.test_hyperparameters._NormalizedDatasetConfig"


class _RegistryStub(_RegistryBase):
    """Minimal _RegistryBase concrete subclass for unit tests."""

    _type_field: ClassVar[str] = "kind"
    _registry: ClassVar[dict] = {}

    kind: str | None = "registry_stub.default"
    value: int = 0


class _RegistryStubWithExtras(_RegistryStub):
    """Subclass with extra fields — used to verify polymorphic-dump preserves
    subclass-specific fields when the parent annotates a base-class slot."""

    extra: str = "subclass-only"


class _NestedModel(BaseModel):
    """Wraps a registry config + a list/dict of registry configs to exercise recursion."""

    pipeline: _RegistryStub
    callbacks: list[_RegistryStub] = []
    by_key: dict[str, _RegistryStub] = {}


class _UnionVariantA(BaseModel):
    """Variant ``A`` of a Pydantic discriminated Union — ``kind`` is a Literal
    with a default, so ``exclude_unset=True`` strips it and reload fails
    without injection."""

    kind: Literal["a"] = "a"
    value_a: int = 0


class _UnionVariantB(BaseModel):
    kind: Literal["b"] = "b"
    value_b: int = 0


_AnyUnion = Annotated[Union[_UnionVariantA, _UnionVariantB], Field(discriminator="kind")]


class _UnionParent(BaseModel):
    """Parent that holds a Pydantic discriminated Union by field."""

    item: _AnyUnion = Field(default_factory=_UnionVariantA)


class _LiteralEnumModel(BaseModel):
    """Parent with a plain ``Literal`` enum field — *not* a Union discriminator.
    These must not be force-injected by the discriminator patcher."""

    accelerator: Literal["cpu", "gpu", "mps"] = "gpu"


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

    def test_injects_pydantic_union_discriminator(self):
        """Union variants identify themselves via a ``Literal[...]`` field with
        a default (e.g. ``FlowMatchingConfig.kind = "flow_matching"``).
        ``exclude_unset=True`` strips it and the dumped dict fails to
        re-validate the union — the patcher must inject it from the live
        instance using the parent field's ``discriminator`` metadata."""
        model = _UnionParent(item=_UnionVariantA(value_a=7))
        dumped = model.model_dump(exclude_unset=True)

        assert "kind" not in dumped["item"]  # baseline: stripped by exclude_unset

        _inject_discriminator_fields(model, dumped)

        assert dumped["item"]["kind"] == "a"

        # Round-trip the patched dump back through the model — without the
        # injected discriminator this would raise a ``union_tag_not_found``.
        reloaded = _UnionParent.model_validate(dumped)
        assert isinstance(reloaded.item, _UnionVariantA)
        assert reloaded.item.value_a == 7

    def test_does_not_inject_plain_literal_enum(self):
        """Plain ``Literal`` enum fields (e.g. ``accelerator: Literal["cpu","gpu","mps"]``)
        are *not* Union discriminators — the patcher must leave them alone so
        ``exclude_unset=True`` semantics are preserved for unrelated enums."""
        model = _LiteralEnumModel()  # accelerator uses default ("gpu")
        dumped = model.model_dump(exclude_unset=True)

        assert "accelerator" not in dumped  # baseline: default-only fields are stripped

        _inject_discriminator_fields(model, dumped)

        assert "accelerator" not in dumped  # still stripped — not a discriminator


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

    def test_load_resolved_round_trip(self, mock_params: MockHyperparameters, tmp_path: Path):
        """``load_resolved`` is the inverse of ``save_resolved`` — the loaded
        config compares equal to what was saved."""
        out_file = tmp_path / "hp.yaml"
        Hyperparameters.save_resolved(mock_params, out_file)

        loaded = Hyperparameters.load_resolved(out_file)

        assert isinstance(loaded, MockHyperparameters)
        assert loaded == mock_params

    def test_load_resolved_resolves_config_schema_kind(self, mock_params: MockHyperparameters, tmp_path: Path):
        """``load_resolved`` returns the concrete ``ConfigSchema`` subclass
        named by the saved ``config_schema_kind`` field, not the base."""
        out_file = tmp_path / "hp.yaml"
        Hyperparameters.save_resolved(mock_params, out_file)

        loaded = Hyperparameters.load_resolved(out_file)

        # The subclass-only ``spec`` field survives — i.e. we got the right class back.
        assert type(loaded) is MockHyperparameters
        assert loaded.spec == mock_params.spec

    def test_load_resolved_missing_file(self, tmp_path: Path):
        """``load_resolved`` raises ``FileNotFoundError`` on a missing path
        (rather than the less-helpful ``yaml`` parse error)."""
        with pytest.raises(FileNotFoundError, match="resolved hyperparameters"):
            Hyperparameters.load_resolved(tmp_path / "does_not_exist.yaml")

    def test_save_resolved_serializes_tensor_stats_as_plain_lists(self, tmp_path: Path):
        """Normalizer stats passed via the config (``TorchTensor`` fields) must be
        written as plain YAML lists, not pickled tensors.

        ``model_dump(serialize_as_any=True)`` duck-types serialization from the
        runtime value and bypasses the ``PlainSerializer`` of the ``TorchTensor``
        annotation, leaving raw ``torch.Tensor`` objects in the dump —
        ``yaml.dump`` then pickles them as ``!!binary`` blobs that neither
        ``yaml.safe_load`` (noether-eval, notebooks) nor ``yaml.full_load``
        (``load_resolved``) can read back.
        """
        os.environ["MASTER_PORT"] = "12345"

        params = MockHyperparameters(
            output_path="/tmp",
            datasets={
                "train": _NormalizedDatasetConfig(
                    dataset_normalizers={"pressure": MeanStdNormalizerConfig(mean=[1.0, 2.0], std=[0.5, 0.25])}
                )
            },
            model=dict(name="abc", kind="xyz"),
            trainer=dict(kind="mock", effective_batch_size=32, callbacks=[], max_epochs=1),
            spec=DimSpec({"def": 1, "abc": 2}),
        )

        out_file = tmp_path / "hp.yaml"
        Hyperparameters.save_resolved(params, out_file)

        raw = out_file.read_text()
        assert "!!binary" not in raw
        assert "!!python/object" not in raw

        # noether-eval and notebooks load with yaml.safe_load — this must not raise.
        with open(out_file) as f:
            content = yaml.safe_load(f)

        normalizer = content["datasets"]["train"]["dataset_normalizers"]["pressure"]
        assert normalizer["mean"] == [1.0, 2.0]
        assert normalizer["std"] == [0.5, 0.25]

    def test_load_resolved_round_trips_tensor_stats(self, tmp_path: Path):
        """``load_resolved`` reads back a config whose normalizer stats came from
        the config (not a STATS_FILE) — and the tensor values survive."""
        os.environ["MASTER_PORT"] = "12345"

        params = MockHyperparameters(
            output_path="/tmp",
            datasets={
                "train": _NormalizedDatasetConfig(
                    dataset_normalizers={"pressure": MeanStdNormalizerConfig(mean=[1.0, 2.0], std=[0.5, 0.25])}
                )
            },
            model=dict(name="abc", kind="xyz"),
            trainer=dict(kind="mock", effective_batch_size=32, callbacks=[], max_epochs=1),
            spec=DimSpec({"def": 1, "abc": 2}),
        )

        out_file = tmp_path / "hp.yaml"
        Hyperparameters.save_resolved(params, out_file)

        loaded = Hyperparameters.load_resolved(out_file)

        normalizer = loaded.datasets["train"].dataset_normalizers["pressure"]
        assert isinstance(normalizer, MeanStdNormalizerConfig)
        assert normalizer.mean.tolist() == [1.0, 2.0]
        assert normalizer.std.tolist() == [0.5, 0.25]

    def test_save_resolved_preserves_polymorphic_subclass_fields(self, tmp_path: Path):
        """When a ``Discriminated`` field is annotated with a base type but
        holds a subclass instance, ``save_resolved`` must preserve the
        subclass-specific fields.

        ``Discriminated`` serializes by runtime class — without it, Pydantic
        walks the *annotated* base type when serializing and silently drops
        fields that exist only on the subclass — e.g.
        ``OfflineLossCallbackConfig.dataset_key`` disappears from a
        ``list[CallBackBaseConfig]`` dump, breaking re-validation.
        """
        os.environ["MASTER_PORT"] = "12345"

        class _PolymorphicHP(ConfigSchema):
            items: list[Annotated[_RegistryStub, Discriminated(_RegistryStub)]] = Field(default_factory=list)

        params = _PolymorphicHP(
            output_path="/tmp",
            datasets=dict(),
            model=dict(name="abc", kind="xyz"),
            trainer=dict(kind="mock", effective_batch_size=32, callbacks=[], max_epochs=1),
            items=[_RegistryStubWithExtras(value=1, extra="kept")],
        )

        out_file = tmp_path / "polymorphic.yaml"
        Hyperparameters.save_resolved(params, out_file)

        with open(out_file) as f:
            content = yaml.safe_load(f)

        # The subclass-only ``extra`` field survives the polymorphic dump.
        assert content["items"][0]["extra"] == "kept"
        # And the discriminator that ``exclude_unset`` would strip is still there.
        assert content["items"][0]["kind"] == "registry_stub.default"
