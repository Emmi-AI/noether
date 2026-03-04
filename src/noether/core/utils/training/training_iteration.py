#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import functools
import operator
import os
import re
from collections.abc import Callable, Generator
from typing import Any, ClassVar


@functools.total_ordering
class TrainingIteration:
    _NAME_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"E(\d+)_U(\d+)_S(\d+)")

    _ITERATION_ATTRIBUTES: ClassVar[tuple[str, ...]] = (
        "epoch",
        "update",
        "sample",
    )  # tuple to keep items ordered and immutable

    def __init__(self, epoch: int | None = None, update: int | None = None, sample: int | None = None):
        self.epoch = epoch
        self.update = update
        self.sample = sample

    def __hash__(self) -> int:
        return hash((self.epoch, self.update, self.sample))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrainingIteration):
            return NotImplemented
        return all(getattr(self, attr_name) == getattr(other, attr_name) for attr_name in self._ITERATION_ATTRIBUTES)

    def __ge__(self, other: TrainingIteration) -> bool:
        if not self.has_same_specified_properties(other):
            raise RuntimeError(f"{self.__class__.__name__} {self} does not have the same properties as {other}")

        for attr_name in self._ITERATION_ATTRIBUTES:
            self_val = getattr(self, attr_name)
            other_val = getattr(other, attr_name)

            if self_val is not None and other_val is not None:
                if self_val < other_val:
                    return False

        return True

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.is_minimally_specified:
            for iter_attr_name in self._ITERATION_ATTRIBUTES:
                self_val = getattr(self, iter_attr_name)
                if self_val is not None:
                    return f"{iter_attr_name.capitalize()}: {self_val}"
        epoch_str = str(int(self.epoch) if isinstance(self.epoch, float) else self.epoch)
        return f"E{epoch_str}_U{self.update}_S{self.sample}"

    def __add__(self, other: TrainingIteration) -> TrainingIteration:
        return self._apply_op(other, operator.add)

    def __sub__(self, other: TrainingIteration) -> TrainingIteration:
        return self._apply_op(other, operator.sub)

    def __iter__(self) -> Generator[tuple[str, int], Any, Any]:
        """Iterates over the properties of the TrainingIteration object and returns them as key-value pairs."""
        for iter_attr_name in self._ITERATION_ATTRIBUTES:
            if getattr(self, iter_attr_name) is not None:
                # with this it will be possible to cast an object to a dict: `dict(MyClass)`
                yield iter_attr_name, getattr(self, iter_attr_name)

    def _apply_op(self, other: TrainingIteration, op: Callable[[int, int], int]) -> TrainingIteration:
        """Helper method to apply an arithmetic operation (+ or -) between two objects."""

        if not self.has_same_specified_properties(other):
            raise RuntimeError(f"{self.__class__.__name__} {self} does not have the same properties as {other}")

        kwargs = {}
        for attr_name in self._ITERATION_ATTRIBUTES:
            other_val = getattr(other, attr_name)
            self_val = getattr(self, attr_name)
            if other_val is not None and self_val is not None:
                kwargs[attr_name] = op(self_val, other_val)

        return TrainingIteration(**kwargs)

    def copy(self) -> TrainingIteration:
        """Creates a copy of a TrainingIteration object."""
        return TrainingIteration(epoch=self.epoch, update=self.update, sample=self.sample)

    @property
    def specified_properties_count(self) -> int:
        """Counts the number of specified (non-None) properties."""
        return sum(1 for attr_name in self._ITERATION_ATTRIBUTES if getattr(self, attr_name) is not None)

    @property
    def is_fully_specified(self) -> bool:
        """Checks if all properties are specified."""
        return self.specified_properties_count == 3

    @property
    def is_minimally_specified(self) -> bool:
        """Checks if at least one property is specified."""
        return self.specified_properties_count == 1

    def get_n_equal_properties(self, other: TrainingIteration) -> int:
        """Counts the number of equal properties between two TrainingIteration objects.

        Args:
            other: The other TrainingIteration object to compare with.
        """
        return sum(getattr(self, key) == getattr(other, key) for key in self._ITERATION_ATTRIBUTES)

    def to_fully_specified(self, updates_per_epoch: int, effective_batch_size: int) -> TrainingIteration:
        """Converts the TrainingIteration object to a fully specified TrainingIteration object.
        This method calculates the missing properties based on the specified ones.

        Args:
            updates_per_epoch: The number of updates per epoch.
            effective_batch_size: The effective batch size used in the training process.
        """

        if self.is_fully_specified:
            return self.copy()

        if not self.is_minimally_specified:
            raise ValueError(
                f"{self.__class__.__name__} '{self}' is not minimally specified, at least one parameter must be set."
            )

        if self.update is not None:
            total_updates = self.update
        elif self.epoch is not None:
            total_updates = updates_per_epoch * self.epoch
        else:
            assert self.sample
            total_updates = int(self.sample / effective_batch_size)

        return TrainingIteration(
            epoch=int(total_updates / updates_per_epoch),
            update=total_updates,
            sample=total_updates * effective_batch_size,
        )

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> TrainingIteration:
        return TrainingIteration(
            epoch=data.get("epoch"),
            update=data.get("update"),
            sample=data.get("sample"),
        )

    def has_same_specified_properties(self, other: TrainingIteration) -> bool:
        """Checks if the specified properties of two TrainingIteration objects are the same.

        Args:
            other: The other TrainingIteration object to compare with.
        """
        return (self.epoch is None, self.update is None, self.sample is None) == (
            other.epoch is None,
            other.update is None,
            other.sample is None,
        )

    @staticmethod
    def from_string(value: str) -> TrainingIteration:
        """Converts a training iteration string into a TrainingIteration object.

        The training iteration string should be in the format "E{epoch}_U{update}_S{sample}".
        For instance: `E5_U12_S123` corresponds to `TrainingIteration(epoch=5, update=12, sample=123)`.

        Args:
            value: The TrainingIteration string to convert."""
        match = TrainingIteration._NAME_PATTERN.match(value)
        if not match:
            raise ValueError(f"Invalid string: {value}")
        epoch, update, sample = map(int, match.groups())
        return TrainingIteration(epoch=epoch, update=update, sample=sample)

    @staticmethod
    def contains_string(source: str) -> bool:
        """Checks if the source string contains a training iteration string.

        Args:
            source: The source string to check.
        Returns:
            bool: True if the source string contains a training iteration string, False otherwise.
        """
        return TrainingIteration._NAME_PATTERN.search(source) is not None

    @staticmethod
    def find_string(source: str) -> str:
        """Checks if the source string contains a training iteration string and returns it.

        Args:
            source: The source string to check.
        Returns:
            str: The found TrainingIteration string.
        """
        match = TrainingIteration._NAME_PATTERN.search(source)
        if not match:
            raise ValueError(f"Could not find TrainingIteration string in '{source}'.")
        return match.group(0)  # group(0) is the full match

    @staticmethod
    def from_filename(fname: str) -> TrainingIteration:
        """Creates a TrainingIteration object from a filename that contains a training iteration string.

        Args:
            fname: The filename to extract the training iteration string from.
        Returns:
            TrainingIteration: The created object."""
        training_iteration_str = TrainingIteration.find_string(fname)
        return TrainingIteration.from_string(training_iteration_str)

    @staticmethod
    def to_fully_specified_from_filenames(
        directory: str,
        training_iteration: TrainingIteration,
        prefix: str | None = None,
        suffix: str | None = None,
    ) -> TrainingIteration:
        """Converts a minimally specified TrainingIteration to a fully specified one
        from filenames in a given folder.
        The first file containing a training iteration string and optionally starting
        with `prefix` or ending with `suffix` is used.

        Args:
            directory: The directory containing training iteration files.
            training_iteration: The TrainingIteration object to match against.
            prefix: Optional prefix to filter files.
            suffix: Optional suffix to filter files.

        Returns:
            TrainingIteration: The created object.

        Raises:
            FileNotFoundError: If no matching file is found.
        """
        if not training_iteration.is_minimally_specified:
            raise ValueError(
                f"TrainingIteration '{training_iteration}' is not minimally specified, at least one parameter "
                "must be set."
            )

        for f in os.listdir(directory):
            # filter irrelevant files
            if prefix is not None and not f.startswith(prefix):
                continue
            if suffix is not None and not f.endswith(suffix):
                continue
            if not TrainingIteration.contains_string(f):
                continue

            instance = TrainingIteration.from_string(TrainingIteration.find_string(f))
            # remove unnecessary properties for comparison (
            # e.g. TrainingIteration(epoch=5, update=12, samples=123) -->
            # TrainingIteration(epoch=5) if training_iteration=TrainingIteration(epoch=123))
            if instance.to_target_specification(training_iteration) == training_iteration:
                return instance
        raise FileNotFoundError(
            "No training iteration file found: "
            f"directory='{directory}', "
            f"training_iteration='{training_iteration}', "
            f"prefix='{prefix}', "
            f"suffix='{suffix}',"
        )

    def to_target_specification(self, target: TrainingIteration) -> TrainingIteration:
        """Creates a new object that matches the target specification.

        Example:
        ```
        self=TrainingIteration(epoch=6, update=12, sample=123)`
        target=TrainingIteration(epoch=5)
        print(self.to_target_specification(target))
        TrainingIteration(epoch=6)
        ```
        Args:
            target: The target TrainingIteration object to match against.

        Returns:
            TrainingIteration: The created object.
        """
        if target.specified_properties_count > self.specified_properties_count:
            raise ValueError(
                f"Target specification count (={target.specified_properties_count}) cannot be greater "
                f"than the source specification (={self.specified_properties_count}). "
                "The target must be a subset of the source's defined properties."
            )
        kwargs = {}

        for iter_attr_name in self._ITERATION_ATTRIBUTES:
            if getattr(target, iter_attr_name) is not None:
                kwargs[iter_attr_name] = getattr(self, iter_attr_name)

        return TrainingIteration(**kwargs)
