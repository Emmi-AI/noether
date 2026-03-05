#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.utils.common.validation import check_inclusive


def test_check_inclusive_all_none():
    assert check_inclusive(None, None) is True


def test_check_inclusive_all_set():
    assert check_inclusive(1, 2) is True


def test_check_inclusive_mixed():
    assert check_inclusive(1, None) is False


def test_check_inclusive_returns_bool():
    result = check_inclusive(None, None)
    assert isinstance(result, bool)
