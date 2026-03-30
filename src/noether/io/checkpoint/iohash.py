#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

# from tempfile import NamedTemporaryFile

DEFAULT_CHUNK = 1024 * 1024  # 1 MiB


def atomic_write_and_hash(
    destination_path: Path,
    chunks: Iterable[bytes],
    *,
    compute_hash: bool = True,
    progress: Any = None,
) -> str | None:
    """Write bytes chunks to a temp file in the same directory as destination_path.
    Additionally, updates SHA-256 while writing, then atomically replaces destination_path.

    Args:
        destination_path: Destination path.
        chunks: Iterable of bytes to write.
        compute_hash: Compute hash of chunks.
        progress: Progress object or None for no progress. Expected to have .update(int) and .close() methods.

    Returns:
        - str: A hex digest.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination_path.with_suffix(destination_path.suffix + ".part")
    hasher = sha256() if compute_hash else None
    try:
        with tmp.open("wb") as f:
            for chunk in chunks:
                if not chunk:
                    continue
                f.write(chunk)
                if hasher:
                    hasher.update(chunk)
                if progress:
                    progress.update(len(chunk))
        tmp.replace(destination_path)
        return hasher.hexdigest() if hasher else None
    finally:
        if progress:
            progress.close()
        if tmp.exists() and not destination_path.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
