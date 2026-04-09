#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Callable, Iterable, Iterator
from typing import Any, Self


class NoopTqdm:
    """A no-operation (noop) version of tqdm that does not display a progress bar."""

    def __init__(self, iterable: Iterable[Any], **kwargs: Any) -> None:
        self.iterable = iterable

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: Any, **__: Any) -> None:
        pass

    def noop(self, *_: Any, **__: Any) -> None:
        pass

    def __getattr__(self, item: Any) -> Callable[[Any], None]:
        return self.noop

    def __iter__(self) -> Iterator[Any]:
        yield from self.iterable
