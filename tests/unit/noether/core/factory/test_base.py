#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from noether.core.factory.base import Factory


class DummyTarget:
    """A dummy class to instantiate."""

    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y


def test_factory_create_none():
    """Test creating None returns None."""
    factory = Factory()
    assert factory.create(None) is None
    assert factory.create({}) is None


def test_factory_create_existing_object():
    """Test passing an already instantiated object returns it as-is."""
    factory = Factory()
    target = DummyTarget()
    assert factory.create(target) is target


def test_factory_create_callable():
    """Test passing a callable (like a partial or class type) instantiates it."""
    factory = Factory()
    # Passing the class type directly -> acts as constructor:
    target = factory.create(DummyTarget, x=5)

    assert isinstance(target, DummyTarget)
    assert target.x == 5


def test_factory_create_from_dict_config_explicit_kind():
    """
    Test instantiating from a dictionary that contains 'kind'.
    We assume 'instantiate' handles the 'kind' lookup.
    """
    factory = Factory()

    # We mock self.instantiate to avoid dealing with the classpath import logic here
    # allowing us to focus on the Factory.create flow:
    with patch.object(factory, "instantiate") as mock_instantiate:
        mock_instantiate.return_value = "created_obj"

        config = {"kind": "some.path", "param": 1}
        result = factory.create(config, extra_arg=2)

        assert result == "created_obj"
        # Verify create() forwarded the dictionary as kwargs to instantiate():
        mock_instantiate.assert_called_with(kind="some.path", param=1, extra_arg=2)


def test_factory_create_partials_mode():
    """Test returns_partials=True behavior."""
    factory = Factory(returns_partials=True)

    # Case 1: Config dict -> returns partial/type (mocked):
    with patch.object(factory, "instantiate") as mock_instantiate:
        mock_type = MagicMock(spec=type)  # pretend to be a class
        mock_instantiate.return_value = mock_type

        result = factory.create({"kind": "foo"})
        assert result == mock_type

    # Case 2: Passing a type/partial directly -> returns it without calling it:
    target_type = DummyTarget
    result = factory.create(target_type)
    # Should NOT instantiate, just return the type:
    assert result == target_type


def test_create_list():
    """Test creating a list of objects."""
    factory = Factory()

    factory.create = MagicMock(side_effect=lambda x, **kwargs: x["val"] if x else None)

    configs = [{"val": 1}, {"val": 2}]
    result = factory.create_list(configs)

    assert result == [1, 2]
    assert factory.create.call_count == 2


def test_create_list_none():
    """Test create_list with None returns empty list."""
    factory = Factory()
    assert factory.create_list(None) == []


def test_create_dict():
    """Test creating a dictionary of objects."""
    factory = Factory()

    factory.create = MagicMock(side_effect=lambda x, **kwargs: x["val"])

    configs = {
        "obj_a": {"val": 10},
        "obj_b": {"val": 20},
    }

    result = factory.create_dict(configs)
    assert result == {"obj_a": 10, "obj_b": 20}


def test_instantiate_real_integration():
    """
    Integration test for `instantiate` using `class_constructor_from_class_path`.
    We rely on reflection_utils working correctly.
    """
    factory = Factory()

    # Use MagicMock because it accepts the positional argument (the config object) that Factory.instantiate passes.
    # types.SimpleNamespace does not support positional args.
    config_obj = SimpleNamespace(kind="unittest.mock.MagicMock")

    # kwargs passed to instantiate should be forwarded (MagicMock sets them as attributes):
    result = factory.instantiate(config_obj, foo="bar")

    assert isinstance(result, MagicMock)
    assert result.foo == "bar"


def test_instantiate_via_kwargs_hack():
    """
    Test the branch: if object_config is None and "kind" in kwargs.
    This is the 'hack' mentioned in your code comments.
    """
    factory = Factory()

    # We pass None as config, but 'kind' in kwargs.
    # In this path, Factory does NOT pass a positional argument, so SimpleNamespace works fine here.
    result = factory.instantiate(None, kind="types.SimpleNamespace", value=123)

    assert isinstance(result, SimpleNamespace)
    assert result.value == 123
