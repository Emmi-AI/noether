#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import TypeVar

KT = TypeVar("KT")
VT = TypeVar("VT")


class Bidict[KT, VT]:
    """A bidirectional dictionary that allows for two-way lookups.

    This dictionary allows you to get values by key and keys by value.
    """

    def __init__(self, forward: dict[KT, VT] | None = None):
        """Create a new Bidict.

        This constructor takes two optional dictionaries for forward and
        backward mapping. At least one of them must be not None.

        Args:
            forward: A dictionary for forward mapping.
            backward: A dictionary for backward mapping.
        """

        self._forward: dict[KT, VT] = {}
        self._backward: dict[VT, KT] = {}

        if forward:
            for key, value in forward.items():
                self.set(key, value)

    def to_forward(self) -> dict[KT, VT]:
        """Return the forward mapping as a dictionary."""
        return self._forward.copy()

    def to_backward(self) -> dict[VT, KT]:
        """Return the backward mapping as a dictionary."""
        return self._backward.copy()

    def get_value_by_key(self, key: KT) -> VT:
        """Get a value by key.

        Args:
            key: The key to look up."""
        return self._forward[key]

    def get_key_by_value(self, value: VT) -> KT:
        """Get a key by value.
        Args:
            key: The value to look up."""
        return self._backward[value]

    def set(self, key: KT, value: VT) -> None:
        """Add a new key-value pair.

        Args:
            key: The key to add.
            value: The value to associate with the key.
        """
        # Clean up stale backward mapping if overwriting an existing key
        if key in self._forward:
            old_value = self._forward[key]
            del self._backward[old_value]
        # Clean up stale forward mapping if overwriting an existing value
        if value in self._backward:
            old_key = self._backward[value]
            del self._forward[old_key]
        self._forward[key] = value
        self._backward[value] = key
