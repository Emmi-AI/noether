#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Annotated, ClassVar

import pytest
from pydantic import BaseModel, Field, ValidationError

from noether.core.schemas.lib import (
    ConfiguredBy,
    Discriminated,
    _discrimnated_validator,
    _RegistryBase,
)

# ---------------------------------------------------------------------------
# Test fixtures: registry roots and config subclasses
#
# NOTE: _RegistryBase inherits from ABC. The __init_subclass__ check
# `issubclass(cls, ABC)` is True for ALL descendants, so auto-registration
# never fires. Tests exercise both the registry lookup path (by manually
# populating the registry) and the dynamic import path (dotted type keys).
# ---------------------------------------------------------------------------


class AnimalConfig(_RegistryBase):
    _registry: ClassVar[dict[str, type[BaseModel]]] = {}
    _type_field: ClassVar[str] = "type"
    type: str | None = None


class DogConfig(AnimalConfig):
    type: str | None = None
    breed: str = "mixed"


class CatConfig(AnimalConfig):
    type: str | None = None
    indoor: bool = True


class ParrotConfig(AnimalConfig):
    type: str | None = None
    can_talk: bool = False
    vocabulary_size: int = 0


class VehicleConfig(_RegistryBase):
    _registry: ClassVar[dict[str, type[BaseModel]]] = {}
    _type_field: ClassVar[str] = "kind"
    kind: str | None = None


class CarConfig(VehicleConfig):
    kind: str | None = None
    doors: int = 4


class BikeConfig(VehicleConfig):
    kind: str | None = None
    electric: bool = False


class StrictConfig(_RegistryBase):
    _registry: ClassVar[dict[str, type[BaseModel]]] = {}
    _type_field: ClassVar[str] = "type"
    type: str | None = None
    model_config = {"extra": "forbid"}


class ConstrainedChild(StrictConfig):
    type: str | None = None
    count: int = Field(..., ge=0)


# Parent models using Discriminated
class SingleAnimalHolder(BaseModel):
    animal: Annotated[AnimalConfig, Discriminated(AnimalConfig)]


class OptionalAnimalHolder(BaseModel):
    animal: Annotated[AnimalConfig, Discriminated(AnimalConfig)] | None = None


class Zoo(BaseModel):
    name: str
    animals: list[Annotated[AnimalConfig, Discriminated(AnimalConfig)]]


class Garage(BaseModel):
    owner: str
    vehicles: list[Annotated[VehicleConfig, Discriminated(VehicleConfig)]]


class AnimalInventory(BaseModel):
    animals: dict[str, Annotated[AnimalConfig, Discriminated(AnimalConfig)]]


def _fqn(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


# Classes for dynamic import tests (must be at module level to be importable)
class _EngineWithCarHint:
    def __init__(self, config: CarConfig):
        pass


@ConfiguredBy(DogConfig)
class _DogModuleWithBaseHint:
    def __init__(self, config: AnimalConfig):
        pass


@ConfiguredBy(CatConfig)
class _PlainCatModule:
    pass


class _SpecialDogConfig(DogConfig):
    special: bool = True


@ConfiguredBy(_SpecialDogConfig)
class _SpecialDogModule:
    def __init__(self, config: DogConfig):
        pass


# ---------------------------------------------------------------------------
# Fixtures to temporarily populate registries
# ---------------------------------------------------------------------------


@pytest.fixture()
def animal_registry():
    AnimalConfig._registry.update({"dog": DogConfig, "cat": CatConfig, "parrot": ParrotConfig})
    yield AnimalConfig._registry
    AnimalConfig._registry.clear()


@pytest.fixture()
def vehicle_registry():
    VehicleConfig._registry.update({"car": CarConfig, "bike": BikeConfig})
    yield VehicleConfig._registry
    VehicleConfig._registry.clear()


@pytest.fixture()
def strict_registry():
    StrictConfig._registry["constrained"] = ConstrainedChild
    yield StrictConfig._registry
    StrictConfig._registry.clear()


# ---------------------------------------------------------------------------
# Tests: _RegistryBase structure
# ---------------------------------------------------------------------------


class TestRegistryBase:
    def test_registry_root_initializes_empty(self):
        assert isinstance(AnimalConfig._registry, dict)

    def test_registries_are_isolated(self):
        assert AnimalConfig._registry is not VehicleConfig._registry

    def test_custom_type_field(self):
        assert VehicleConfig._type_field == "kind"
        assert AnimalConfig._type_field == "type"

    def test_subclass_inherits_type_field(self):
        assert DogConfig._type_field == "type"
        assert CarConfig._type_field == "kind"

    def test_all_subclasses_are_abc_subclasses(self):
        """Since _RegistryBase inherits ABC, issubclass(cls, ABC) is always True,
        preventing auto-registration via __init_subclass__."""
        from abc import ABC

        assert issubclass(DogConfig, ABC)
        assert issubclass(CarConfig, ABC)

    def test_registry_empty_without_manual_population(self):
        class FreshRoot(_RegistryBase):
            _registry: ClassVar[dict[str, type[BaseModel]]] = {}
            type: str | None = None

        class FreshChild(FreshRoot):
            type: str | None = None
            value: int = 1

        assert FreshRoot._registry == {}


class TestRegistryBaseInstantiation:
    def test_instantiate_subclass(self):
        dog = DogConfig(breed="labrador")
        assert dog.breed == "labrador"

    def test_default_values(self):
        cat = CatConfig()
        assert cat.indoor is True

    def test_model_validate_from_dict(self):
        parrot = ParrotConfig.model_validate({"can_talk": True, "vocabulary_size": 50})
        assert parrot.can_talk is True
        assert parrot.vocabulary_size == 50


# ---------------------------------------------------------------------------
# Tests: _discrimnated_validator — passthrough behavior
# ---------------------------------------------------------------------------


class TestDiscriminatedValidatorPassthrough:
    def test_model_instance_passthrough(self):
        dog = DogConfig(breed="husky")
        result = _discrimnated_validator(dog, registry_cls=AnimalConfig)
        assert result is dog

    def test_string_passthrough(self):
        assert _discrimnated_validator("hello", registry_cls=AnimalConfig) == "hello"

    def test_none_passthrough(self):
        assert _discrimnated_validator(None, registry_cls=AnimalConfig) is None

    def test_int_passthrough(self):
        assert _discrimnated_validator(42, registry_cls=AnimalConfig) == 42

    def test_list_passthrough(self):
        assert _discrimnated_validator([1, 2], registry_cls=AnimalConfig) == [1, 2]

    def test_dict_without_type_field_passthrough(self):
        data = {"breed": "poodle"}
        assert _discrimnated_validator(data, registry_cls=AnimalConfig) == data

    def test_empty_dict_passthrough(self):
        assert _discrimnated_validator({}, registry_cls=AnimalConfig) == {}

    def test_none_type_value_raises_type_error(self):
        """type=None triggers `"." in type_key` which raises TypeError."""
        with pytest.raises(TypeError):
            _discrimnated_validator({"type": None}, registry_cls=AnimalConfig)

    def test_unknown_key_without_dot_passthrough(self):
        data = {"type": "unknown_animal"}
        assert _discrimnated_validator(data, registry_cls=AnimalConfig) == data


# ---------------------------------------------------------------------------
# Tests: _discrimnated_validator — registry lookup
# ---------------------------------------------------------------------------


class TestDiscriminatedValidatorRegistryLookup:
    def test_lookup_dog(self, animal_registry):
        result = _discrimnated_validator({"type": "dog", "breed": "poodle"}, registry_cls=AnimalConfig)
        assert isinstance(result, DogConfig)
        assert result.breed == "poodle"

    def test_lookup_cat(self, animal_registry):
        result = _discrimnated_validator({"type": "cat", "indoor": False}, registry_cls=AnimalConfig)
        assert isinstance(result, CatConfig)
        assert result.indoor is False

    def test_lookup_parrot(self, animal_registry):
        result = _discrimnated_validator(
            {"type": "parrot", "can_talk": True, "vocabulary_size": 50},
            registry_cls=AnimalConfig,
        )
        assert isinstance(result, ParrotConfig)
        assert result.can_talk is True
        assert result.vocabulary_size == 50

    def test_lookup_with_custom_type_field(self, vehicle_registry):
        result = _discrimnated_validator({"kind": "car", "doors": 2}, registry_cls=VehicleConfig)
        assert isinstance(result, CarConfig)
        assert result.doors == 2

    def test_lookup_bike(self, vehicle_registry):
        result = _discrimnated_validator({"kind": "bike", "electric": True}, registry_cls=VehicleConfig)
        assert isinstance(result, BikeConfig)
        assert result.electric is True

    def test_registry_takes_precedence_over_dynamic_import(self, animal_registry):
        AnimalConfig._registry["some.dotted.key"] = DogConfig
        result = _discrimnated_validator(
            {"type": "some.dotted.key", "breed": "retriever"},
            registry_cls=AnimalConfig,
        )
        assert isinstance(result, DogConfig)
        assert result.breed == "retriever"

    def test_lookup_with_defaults(self, animal_registry):
        result = _discrimnated_validator({"type": "dog"}, registry_cls=AnimalConfig)
        assert isinstance(result, DogConfig)
        assert result.breed == "mixed"

    def test_unknown_key_falls_through(self, animal_registry):
        data = {"type": "fish"}
        assert _discrimnated_validator(data, registry_cls=AnimalConfig) == data


# ---------------------------------------------------------------------------
# Tests: _discrimnated_validator — dynamic import path
# ---------------------------------------------------------------------------


class TestDiscriminatedValidatorDynamicImport:
    def test_import_registry_subclass(self):
        result = _discrimnated_validator({"type": _fqn(DogConfig), "breed": "poodle"}, registry_cls=AnimalConfig)
        assert isinstance(result, DogConfig)
        assert result.breed == "poodle"

    def test_import_another_subclass(self):
        result = _discrimnated_validator({"type": _fqn(CatConfig), "indoor": False}, registry_cls=AnimalConfig)
        assert isinstance(result, CatConfig)
        assert result.indoor is False

    def test_import_with_custom_type_field(self):
        result = _discrimnated_validator({"kind": _fqn(CarConfig), "doors": 2}, registry_cls=VehicleConfig)
        assert isinstance(result, CarConfig)
        assert result.doors == 2

    def test_import_with_extra_data(self):
        result = _discrimnated_validator(
            {"type": _fqn(ParrotConfig), "can_talk": True, "vocabulary_size": 100},
            registry_cls=AnimalConfig,
        )
        assert isinstance(result, ParrotConfig)
        assert result.vocabulary_size == 100

    def test_nonexistent_module_raises(self):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            _discrimnated_validator({"type": "nonexistent.module.ClassName"}, registry_cls=AnimalConfig)

    def test_nonexistent_attr_raises(self):
        with pytest.raises(AttributeError):
            _discrimnated_validator({"type": "pydantic.NonExistentClass"}, registry_cls=AnimalConfig)

    def test_non_subclass_without_config_raises(self):
        with pytest.raises(ValueError, match="Unknown type key"):
            _discrimnated_validator({"type": "pydantic.BaseModel"}, registry_cls=AnimalConfig)

    def test_resolve_via_init_type_hint(self):
        result = _discrimnated_validator({"kind": _fqn(_EngineWithCarHint), "doors": 6}, registry_cls=VehicleConfig)
        assert isinstance(result, CarConfig)
        assert result.doors == 6

    def test_import_with_default_values(self):
        result = _discrimnated_validator({"type": _fqn(DogConfig)}, registry_cls=AnimalConfig)
        assert isinstance(result, DogConfig)
        assert result.breed == "mixed"


# ---------------------------------------------------------------------------
# Tests: ConfiguredBy decorator
# ---------------------------------------------------------------------------


class TestConfiguredBy:
    def test_sets_config_class_attr(self):
        @ConfiguredBy(DogConfig)
        class MyClass:
            pass

        assert MyClass._config_class is DogConfig

    def test_preserves_class_identity(self):
        @ConfiguredBy(DogConfig)
        class Original:
            x = 42

        assert Original.x == 42
        assert Original.__name__ == "Original"

    def test_resolution_via_validator(self):
        result = _discrimnated_validator(
            {"type": _fqn(_DogModuleWithBaseHint), "breed": "shiba"}, registry_cls=AnimalConfig
        )
        assert isinstance(result, DogConfig)
        assert result.breed == "shiba"

    def test_resolution_without_init_hint(self):
        result = _discrimnated_validator({"type": _fqn(_PlainCatModule), "indoor": False}, registry_cls=AnimalConfig)
        assert isinstance(result, CatConfig)
        assert result.indoor is False

    def test_config_class_takes_precedence_when_more_specific(self):
        result = _discrimnated_validator(
            {"type": _fqn(_SpecialDogModule), "breed": "x", "special": True},
            registry_cls=AnimalConfig,
        )
        assert isinstance(result, _SpecialDogConfig)


# ---------------------------------------------------------------------------
# Tests: Discriminated (BeforeValidator) integrated with Pydantic models
# ---------------------------------------------------------------------------


class TestDiscriminatedIntegration:
    def test_single_field_via_dynamic_import(self):
        holder = SingleAnimalHolder.model_validate({"animal": {"type": _fqn(DogConfig), "breed": "beagle"}})
        assert isinstance(holder.animal, DogConfig)
        assert holder.animal.breed == "beagle"

    def test_single_field_via_registry(self, animal_registry):
        holder = SingleAnimalHolder.model_validate({"animal": {"type": "cat", "indoor": False}})
        assert isinstance(holder.animal, CatConfig)
        assert holder.animal.indoor is False

    def test_list_of_mixed_animals_via_import(self):
        zoo = Zoo.model_validate(
            {
                "name": "City Zoo",
                "animals": [
                    {"type": _fqn(DogConfig), "breed": "dalmatian"},
                    {"type": _fqn(CatConfig)},
                    {"type": _fqn(ParrotConfig), "can_talk": True, "vocabulary_size": 100},
                ],
            }
        )
        assert len(zoo.animals) == 3
        assert isinstance(zoo.animals[0], DogConfig)
        assert isinstance(zoo.animals[1], CatConfig)
        assert isinstance(zoo.animals[2], ParrotConfig)
        assert zoo.animals[2].vocabulary_size == 100

    def test_list_of_mixed_animals_via_registry(self, animal_registry):
        zoo = Zoo.model_validate(
            {
                "name": "City Zoo",
                "animals": [
                    {"type": "dog", "breed": "dalmatian"},
                    {"type": "cat"},
                    {"type": "parrot", "can_talk": True, "vocabulary_size": 100},
                ],
            }
        )
        assert len(zoo.animals) == 3
        assert isinstance(zoo.animals[0], DogConfig)
        assert isinstance(zoo.animals[1], CatConfig)
        assert isinstance(zoo.animals[2], ParrotConfig)

    def test_empty_list(self):
        zoo = Zoo.model_validate({"name": "Empty Zoo", "animals": []})
        assert zoo.animals == []

    def test_dict_values(self):
        inventory = AnimalInventory.model_validate(
            {
                "animals": {
                    "buddy": {"type": _fqn(DogConfig), "breed": "golden"},
                    "whiskers": {"type": _fqn(CatConfig), "indoor": True},
                }
            }
        )
        assert isinstance(inventory.animals["buddy"], DogConfig)
        assert isinstance(inventory.animals["whiskers"], CatConfig)

    def test_optional_field_none(self):
        holder = OptionalAnimalHolder.model_validate({"animal": None})
        assert holder.animal is None

    def test_optional_field_present(self):
        holder = OptionalAnimalHolder.model_validate({"animal": {"type": _fqn(DogConfig)}})
        assert isinstance(holder.animal, DogConfig)

    def test_optional_field_default(self):
        holder = OptionalAnimalHolder.model_validate({})
        assert holder.animal is None

    def test_already_instantiated_passthrough(self):
        dog = DogConfig(breed="pug")
        holder = SingleAnimalHolder.model_validate({"animal": dog})
        assert holder.animal is dog

    def test_vehicle_list(self):
        garage = Garage.model_validate(
            {
                "owner": "Alice",
                "vehicles": [
                    {"kind": _fqn(CarConfig), "doors": 2},
                    {"kind": _fqn(BikeConfig), "electric": True},
                ],
            }
        )
        assert isinstance(garage.vehicles[0], CarConfig)
        assert garage.vehicles[0].doors == 2
        assert isinstance(garage.vehicles[1], BikeConfig)
        assert garage.vehicles[1].electric is True

    def test_multiple_registries_in_one_model(self):
        class PetOwner(BaseModel):
            pet: Annotated[AnimalConfig, Discriminated(AnimalConfig)]
            vehicle: Annotated[VehicleConfig, Discriminated(VehicleConfig)]

        owner = PetOwner.model_validate(
            {
                "pet": {"type": _fqn(CatConfig), "indoor": True},
                "vehicle": {"kind": _fqn(BikeConfig), "electric": True},
            }
        )
        assert isinstance(owner.pet, CatConfig)
        assert isinstance(owner.vehicle, BikeConfig)


# ---------------------------------------------------------------------------
# Tests: Validation error propagation
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_invalid_field_type_via_registry(self, animal_registry):
        with pytest.raises(ValidationError):
            _discrimnated_validator(
                {"type": "parrot", "vocabulary_size": "abc"},
                registry_cls=AnimalConfig,
            )

    def test_invalid_field_type_via_import(self):
        with pytest.raises(ValidationError):
            _discrimnated_validator(
                {"type": _fqn(ParrotConfig), "vocabulary_size": "not_an_int"},
                registry_cls=AnimalConfig,
            )

    def test_constraint_violation_via_registry(self, strict_registry):
        with pytest.raises(ValidationError):
            _discrimnated_validator({"type": "constrained", "count": -1}, registry_cls=StrictConfig)

    def test_constraint_violation_via_import(self):
        with pytest.raises(ValidationError):
            _discrimnated_validator({"type": _fqn(ConstrainedChild), "count": -1}, registry_cls=StrictConfig)

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            _discrimnated_validator({"type": _fqn(ConstrainedChild)}, registry_cls=StrictConfig)

    def test_extra_fields_rejected(self, strict_registry):
        with pytest.raises(ValidationError):
            _discrimnated_validator(
                {"type": "constrained", "count": 5, "unknown": 1},
                registry_cls=StrictConfig,
            )


# ---------------------------------------------------------------------------
# Tests: Roundtrip serialization
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_model_dump_via_registry(self, animal_registry):
        original = DogConfig(type="dog", breed="corgi")
        dumped = original.model_dump()
        restored = _discrimnated_validator(dumped, registry_cls=AnimalConfig)
        assert isinstance(restored, DogConfig)
        assert restored.breed == "corgi"

    def test_model_dump_via_import(self):
        fqn = _fqn(DogConfig)
        original = DogConfig(type=fqn, breed="corgi")
        dumped = original.model_dump()
        restored = _discrimnated_validator(dumped, registry_cls=AnimalConfig)
        assert isinstance(restored, DogConfig)
        assert restored.breed == "corgi"

    def test_json_roundtrip_preserves_type_via_import(self):
        """Pydantic serializes based on the annotated type (AnimalConfig), so
        subclass-specific fields are lost in model_dump. But the type field
        roundtrips correctly, allowing re-resolution to the right subclass."""
        zoo = Zoo.model_validate(
            {
                "name": "Test Zoo",
                "animals": [
                    {"type": _fqn(DogConfig), "breed": "shiba"},
                    {"type": _fqn(CatConfig), "indoor": False},
                ],
            }
        )
        json_str = zoo.model_dump_json()
        restored = Zoo.model_validate_json(json_str)
        assert len(restored.animals) == 2
        assert isinstance(restored.animals[0], DogConfig)
        assert isinstance(restored.animals[1], CatConfig)

    def test_json_roundtrip_via_registry(self, animal_registry):
        zoo = Zoo.model_validate(
            {
                "name": "Zoo",
                "animals": [
                    {"type": "dog", "breed": "lab"},
                    {"type": "cat"},
                ],
            }
        )
        json_str = zoo.model_dump_json()
        restored = Zoo.model_validate_json(json_str)
        assert isinstance(restored.animals[0], DogConfig)
        assert isinstance(restored.animals[1], CatConfig)
