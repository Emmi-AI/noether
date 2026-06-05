#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Storage helpers so the Zarr store can live locally or on object storage.

A ``store_root`` may be a local path or an fsspec URL (``s3://``, ``gs://``, ``az://``,
``memory://``, …). Local roots use the fast :class:`~zarr.storage.LocalStore`; URLs use
:class:`~zarr.storage.FsspecStore`. Path joining is done as plain ``/``-joins so it works
for both local paths and URLs (``pathlib`` would mangle ``s3://`` into ``s3:/``).

For ``s3://`` URLs, if the optional `obstore <https://developmentseed.org/obstore/>`_
package is installed, the faster Rust-backed :class:`~zarr.storage.ObjectStore` is used
instead of :class:`~zarr.storage.FsspecStore` (it reads credentials/region/endpoint from
the standard ``AWS_*`` environment variables). Without obstore, or for any non-S3 URL,
the fsspec backend is used — so this is a transparent, opt-in speedup.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from zarr.abc.store import Store
from zarr.storage import FsspecStore, LocalStore

_FILE_PREFIX = "file://"
_S3_PREFIX = "s3://"


def _obstore_available() -> bool:
    """Return True if the optional ``obstore`` package can be imported."""
    return importlib.util.find_spec("obstore") is not None


def is_remote(store_root: str | Path) -> bool:
    """Return True if *store_root* is an fsspec URL backed by a non-local filesystem."""
    text = str(store_root)
    return "://" in text and not text.startswith(_FILE_PREFIX)


def join(store_root: str | Path, relpath: str) -> str:
    """Join *relpath* onto a local path or URL store root (URL-safe, unlike ``Path``)."""
    return f"{str(store_root).rstrip('/')}/{relpath}"


def _make_obstore_s3(url: str, *, read_only: bool) -> Store:
    """Build an obstore-backed :class:`ObjectStore` for an ``s3://`` URL.

    Credentials, region and endpoint are read from the standard ``AWS_*`` environment
    variables at request time (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
    ``AWS_REGION``/``AWS_DEFAULT_REGION``, ``AWS_ENDPOINT_URL``, …).
    """
    from obstore.store import S3Store
    from zarr.storage import ObjectStore

    return ObjectStore(S3Store.from_url(url), read_only=read_only)


def make_store(path: str | Path, *, read_only: bool = False) -> Store:
    """Build a Zarr store for *path*.

    Uses the Rust-backed :class:`~zarr.storage.ObjectStore` for ``s3://`` URLs when the
    optional ``obstore`` package is installed, :class:`FsspecStore` for any other URL (or
    for S3 without obstore), and :class:`LocalStore` for local paths.
    """
    text = str(path)
    if text.startswith(_S3_PREFIX) and _obstore_available():
        return _make_obstore_s3(text, read_only=read_only)
    if is_remote(text):
        return FsspecStore.from_url(text, read_only=read_only)
    text = text.removeprefix(_FILE_PREFIX)
    return LocalStore(text, read_only=read_only)
