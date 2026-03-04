#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest

from noether.core.utils.bidict import Bidict


def test_init_empty():
    """Test initializing an empty Bidict."""
    b = Bidict()
    assert b.to_forward() == {}
    assert b.to_backward() == {}


def test_init_with_data():
    """Test initializing with a forward dictionary."""
    data = {"apple": 1, "banana": 2}
    b = Bidict(forward=data)

    # Check forward lookups:
    assert b.get_value_by_key("apple") == 1
    assert b.get_value_by_key("banana") == 2

    # Check backward lookups (automatically generated):
    assert b.get_key_by_value(1) == "apple"
    assert b.get_key_by_value(2) == "banana"


def test_set_and_get():
    """Test manually setting items and retrieving them both ways."""
    b = Bidict[str, str]()
    b.set("k1", "v1")

    assert b.get_value_by_key("k1") == "v1"
    assert b.get_key_by_value("v1") == "k1"


def test_missing_keys():
    """Test that standard KeyError is raised for missing items."""
    b = Bidict()

    with pytest.raises(KeyError):
        b.get_value_by_key("ghost")

    with pytest.raises(KeyError):
        b.get_key_by_value("ghost")


def test_update_existing_key():
    """Test updating the value for an existing key cleans up stale backward mapping."""
    b = Bidict({"k": "v_old"})

    # Update k to point to v_new:
    b.set("k", "v_new")

    # Forward map should return new value:
    assert b.get_value_by_key("k") == "v_new"

    # Backward map should have the new mapping:
    assert b.get_key_by_value("v_new") == "k"

    # Old backward link should be cleaned up:
    with pytest.raises(KeyError):
        b.get_key_by_value("v_old")


def test_update_existing_value():
    """Test that reassigning a value to a new key cleans up stale forward mapping."""
    b = Bidict({"k_old": "v"})

    # Reassign value "v" to a new key "k_new":
    b.set("k_new", "v")

    # Backward should point to the new key:
    assert b.get_key_by_value("v") == "k_new"
    assert b.get_value_by_key("k_new") == "v"

    # Old forward link should be cleaned up:
    with pytest.raises(KeyError):
        b.get_value_by_key("k_old")


def test_to_dictionaries_are_copies():
    """Verify that to_forward() and to_backward() return copies, not references."""
    b = Bidict({"a": 1})

    fwd = b.to_forward()
    bwd = b.to_backward()

    # Modify the returned dicts:
    fwd["b"] = 2
    bwd[2] = "b"

    # Ensure the Bidict internal state is untouched:
    with pytest.raises(KeyError):
        b.get_value_by_key("b")

    with pytest.raises(KeyError):
        b.get_key_by_value(2)
