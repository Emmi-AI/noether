#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Storage helpers so the Zarr store can live locally or on object storage.

A ``store_root`` may be a local path or an fsspec URL (``s3://``, ``gs://``, ``az://``,
``memory://``, …). Local roots use the fast :class:`~zarr.storage.LocalStore`; URLs use
:class:`~zarr.storage.FsspecStore`. Path joining is done as plain ``/``-joins so it works
for both local paths and URLs (``pathlib`` would mangle ``s3://`` into ``s3:/``).
"""

from __future__ import annotations

from pathlib import Path

from zarr.abc.store import Store
from zarr.storage import FsspecStore, LocalStore

_FILE_PREFIX = "file://"


def is_remote(store_root: str | Path) -> bool:
    """Return True if *store_root* is an fsspec URL backed by a non-local filesystem."""
    text = str(store_root)
    return "://" in text and not text.startswith(_FILE_PREFIX)


def join(store_root: str | Path, relpath: str) -> str:
    """Join *relpath* onto a local path or URL store root (URL-safe, unlike ``Path``)."""
    return f"{str(store_root).rstrip('/')}/{relpath}"


def make_store(path: str | Path, *, read_only: bool = False) -> Store:
    """Build a Zarr store for *path*: :class:`FsspecStore` for URLs, else :class:`LocalStore`."""
    text = str(path)
    if is_remote(text):
        return FsspecStore.from_url(text, read_only=read_only)
    text = text.removeprefix(_FILE_PREFIX)
    return LocalStore(text, read_only=read_only)
