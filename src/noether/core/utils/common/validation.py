#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any


def float_to_integer_exact(f: float) -> int:
    """Converts floats without decimals to int."""
    assert f.is_integer()
    return int(f)


def check_exclusive(*args: Any) -> bool:
    """Checks if exactly one of the arguments is not None."""
    return sum(arg is not None for arg in args) == 1


def check_inclusive(*args: Any) -> bool:
    """Checks if either all arguments are not None or if all are None."""
    return sum(arg is not None for arg in args) in [0, len(args)]


def check_at_least_one(*args: Any) -> bool:
    """Checks if at least one of the arguments is not None."""
    return sum(arg is not None for arg in args) > 0


def check_at_most_one(*args: Any) -> bool:
    """Checks if at most one of the arguments is not None."""
    return sum(arg is not None for arg in args) <= 1


def check_all_none(*args: Any) -> bool:
    """Checks if all arguments are None."""
    return sum(arg is not None for arg in args) == 0
