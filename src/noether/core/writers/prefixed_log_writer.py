#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from noether.core.writers.log_writer import LogWriter


class PrefixedLogWriter:
    """Proxy over :class:`LogWriter` that prepends a prefix to every logged key.

    Used by composite callbacks (e.g. :class:`~noether.core.callbacks.checkpoint.ema.EmaCallback`) that run
    child evaluation callbacks under alternate model weights, so that the child's metric keys don't collide
    with the live-model metrics. All non-logging methods (``flush``, ``__enter__``/``__exit__``, etc.) are
    delegated to the wrapped writer so the underlying cache/history stay consistent.

    Args:
        inner: The underlying :class:`LogWriter` to delegate to.
        prefix: Prefix to prepend to every key (trailing slashes are stripped).
    """

    def __init__(self, inner: LogWriter, prefix: str) -> None:
        self._inner = inner
        self._prefix = prefix.rstrip("/")

    def _prefixed(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def add_scalar(
        self,
        key: str,
        value: torch.Tensor | np.generic | float,
        logger: logging.Logger | None = None,
        format_str: str | None = None,
    ) -> None:
        """Forward to the underlying writer with the key prefixed."""
        self._inner.add_scalar(self._prefixed(key), value, logger=logger, format_str=format_str)

    def add_nonscalar(self, key: str, value: Any) -> None:
        """Forward to the underlying writer with the key prefixed."""
        self._inner.add_nonscalar(self._prefixed(key), value)

    def get_all_metric_values(self, key: str) -> list[float]:
        """Look up values under the prefixed key."""
        return self._inner.get_all_metric_values(self._prefixed(key))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
