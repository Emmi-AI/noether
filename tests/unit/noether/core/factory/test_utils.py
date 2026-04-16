#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest

from noether.core.factory.utils import all_ctor_kwarg_names, class_constructor_from_class_path


class Parent:
    def __init__(self, p_arg, p_kwarg="default"):
        pass


class Child(Parent):
    def __init__(self, c_arg, c_kwarg=None, **kwargs):
        super().__init__(p_arg="test", **kwargs)


class GrandChild(Child):
    pass


def test_all_ctor_kwarg_names_simple():
    """Test extracting args from a single class."""
    args = all_ctor_kwarg_names(Parent)
    assert "p_arg" in args
    assert "p_kwarg" in args


def test_all_ctor_kwarg_names_inheritance():
    """Test extracting args traversing the MRO."""
    args = all_ctor_kwarg_names(Child)

    # Should have Child's args:
    assert "c_arg" in args
    assert "c_kwarg" in args
    # Should have Parent's args:
    assert "p_arg" in args
    assert "p_kwarg" in args


def test_all_ctor_kwarg_names_empty_inheritance():
    """Test class that inherits but doesn't override init."""
    args = all_ctor_kwarg_names(GrandChild)
    assert "c_arg" in args
    assert "p_arg" in args


def test_class_constructor_valid():
    """Test importing a valid standard library class."""
    # 'collections.OrderedDict'
    constructor = class_constructor_from_class_path("collections.OrderedDict")

    from collections import OrderedDict

    assert constructor is OrderedDict

    # Instantiate to be sure:
    instance = constructor([(1, 2)])
    assert isinstance(instance, OrderedDict)
    assert instance[1] == 2


def test_class_constructor_invalid_module():
    """Test error when module doesn't exist."""
    with pytest.raises(ModuleNotFoundError):
        class_constructor_from_class_path("fake_module_xyz.MyClass")


def test_class_constructor_invalid_class():
    """Test error when module exists but class doesn't."""
    with pytest.raises(AttributeError):
        class_constructor_from_class_path("collections.FakeClassXYZ")


def test_class_constructor_bad_format():
    """Test assertions for invalid strings."""
    with pytest.raises(ValueError):
        class_constructor_from_class_path("JustClassNameWithoutModule")
