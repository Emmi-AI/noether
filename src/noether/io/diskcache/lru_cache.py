#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""LRU cache filesystem implementations with size management and cleanup.

This module provides filesystem cache implementations that wrap other filesystems
(local or remote) and maintain a local cache of accessed files. The cache is
managed using a Least Recently Used (LRU) eviction policy to stay within
configurable size limits.

The module includes:
    - LRUCacheFileSystem: A basic LRU cache implementation using filesystem metadata.
    - SqliteLRUCacheFileSystem: An enhanced LRU cache using SQLite for metadata tracking,
      suitable for concurrent access scenarios.

Key Features:
    - Automatic cache size management with configurable watermarks
    - Thread-safe file locking to prevent race conditions
    - LRU-based eviction of cached files when size limits are exceeded
    - Support for any fsspec-compatible filesystem as the backing store
    - SQLite-based metadata tracking for improved concurrency

Example:
    Basic usage with an S3 filesystem::


        fs = LRUCacheFileSystem(
            cache_size=10**9,  # 1 GB limit

        with fs.open("s3://my-bucket/data.csv", "rb") as f:
            data = f.read() # standard file-like object

Note:
    The LRU tracking updates file access times on every read, which may impact
    performance on some filesystems. The SqliteLRUCacheFileSystem is recommended
    for multi-threaded or concurrent access patterns.

Attributes:
    LOCK_FILE_MODE (int): Default file mode for lock files (0o660).
    LOCK_FILE_SUFFIX (str): Suffix appended to cached file paths for lock files.
    CLEANUP_LOCK_FILE (str): Name of the global cleanup lock file.
"""

import contextlib
import inspect
import logging
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Literal

import tenacity
from filelock import FileLock, Timeout
from fsspec import AbstractFileSystem, filesystem  # type: ignore[import-untyped]
from fsspec.implementations.cached import CachingFileSystem  # type: ignore[import-untyped]
from fsspec.implementations.local import LocalFileSystem  # type: ignore[import-untyped]

logger = logging.getLogger("capped_cache")

LOCK_FILE_MODE = 0o660
LOCK_FILE_SUFFIX = ".lock"
CLEANUP_LOCK_FILE = ".cleanup_lock"


class _MultiFileLock:
    """
    A context manager for acquiring multiple FileLock objects at once.
    """

    def __init__(self, file_paths: list[str], timeout: float = -1):
        """
        Initializes the MultiFileLock.

        Args:
            file_paths: A list of file paths to lock.
            timeout: The maximum time in seconds to wait for the locks.
                             A value of -1 means wait indefinitely.
        """
        self._locks = [FileLock(path + LOCK_FILE_SUFFIX, timeout=timeout, mode=LOCK_FILE_MODE) for path in file_paths]

    def __enter__(self):
        """
        Acquires all file locks in the order they were provided.
        If a lock cannot be acquired, it releases any previously acquired locks.
        """
        acquired_locks = []
        try:
            for lock in self._locks:
                lock.acquire()
                acquired_locks.append(lock)
        except Timeout:
            # If a timeout occurs, release any locks that were successfully acquired
            logger.info("A timeout occurred while trying to acquire a lock.")
            for acquired_lock in reversed(acquired_locks):
                acquired_lock.release()
            raise  # Re-raise the Timeout exception
        except Exception as e:
            # Handle other potential exceptions during acquisition
            logger.error(f"An error occurred: {e}")
            for acquired_lock in reversed(acquired_locks):
                acquired_lock.release()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Releases all acquired file locks in the reverse order of acquisition.
        """
        for lock in reversed(self._locks):
            if lock.is_locked:
                lock.release()


def _cache_mapper(path: str) -> str:
    return "_@_".join(path.split("/"))


class LRUCacheFileSystem(CachingFileSystem):
    """
    Caches whole remote files on first access, with LRU cache eviction.
    This class is intended as a layer over any other file system, and
    will make a local copy of each file accessed, so that all subsequent
    reads are local. This implementation only copies whole files.

    The cache is kept within a specified size limit by removing the least
    recently used files when the limit is exceeded. File access times
    are updated on each access to ensure accurate LRU tracking.
    **NOTE** every file access requires a write to the filesystem to
    update the access time, which may lead to performance issues on
    some filesystems.
    See `fsspec.implementations.cached.SimpleCacheFileSystem` for a simpler implementation that
    does not delete old files.

    Examples:
        .. code-block:: python

            import fsspec
            from noether.io.diskcache.lru_cache import LRUCacheFileSystem

            fs = LRUCacheFileSystem(
                fs=fsspec.filesystem("s3"),
                storage="/tmp/cache",
                cache_storage_size=10**9,  # 1 GB
            )
            with fs.open("s3://my-bucket/my-large-file.dat", "rb") as f:
                data = f.read()
    """

    protocol = "lrucache"
    local_file = True

    def __init__(
        self,
        storage: str,
        cache_size: int | None,
        target_protocol: str | None = None,
        target_options: dict | None = None,
        fs: AbstractFileSystem | None = None,
        cache_storage_dmode=0o770,
        cache_storage_mode=0o660,
        enforce_size_every_seconds=2,
        cache_cleanup_high_watermark=0.95,
        cache_cleanup_low_watermark=0.8,
        **kwargs,
    ):
        """
        Args:
            fs: The target filesystem to wrap with caching.
            storage: Path to the local directory where cached files will be stored.
            cache_size: Maximum size of the cache in bytes. If None, no size limit is enforced.
            cache_storage_dmode: Directory permissions mode for the cache storage directory.
            cache_storage_mode: File permissions mode for cached files.
            enforce_size_every_seconds: Minimum time in seconds between cache size enforcement checks.
            cache_cleanup_high_watermark: Fraction of cache_size at which cleanup is triggered.
            cache_cleanup_low_watermark: Target fraction of cache_size to reduce to during cleanup.

        """
        if target_options is None:
            target_options = {}
        super().__init__(**{**kwargs, "target_protocol": target_protocol, "target_options": target_options, "fs": fs})
        self.fs = fs if fs is not None else filesystem(target_protocol, **(target_options or {}))  # type: ignore
        self.storage = storage
        self.cache_storage_size = cache_size
        self.cache_storage_mode = cache_storage_mode
        self.enforce_size_every_seconds = enforce_size_every_seconds
        self._last_enforce_time = 0.0
        self.cache_cleanup_high_watermark = cache_cleanup_high_watermark
        self.cache_cleanup_low_watermark = cache_cleanup_low_watermark

        if not os.path.exists(storage):
            os.makedirs(storage, exist_ok=True)
            os.chmod(storage, cache_storage_dmode)
        self._local = LocalFileSystem()
        self._mapper = _cache_mapper

        def _strip_protocol(path):
            # acts as a method, since each instance has a difference target
            return self.fs._strip_protocol(type(self)._strip_protocol(path))

        self._strip_protocol = _strip_protocol

    def __getattribute__(self, item):
        if item in {
            "_open",
            "__init__",
            "__getattribute__",
            "__reduce__",
            "_atomic_get",
            "open",
            "cat",
            "cat_file",
            "_cat_file",
            "cat_ranges",
            "_cat_ranges",
            "get",
            "_check_file",
            "_check_cache",
            "_mkcache",
            "__hash__",
            "__eq__",
            "_ensure_schema",
            "_enforce_cache_size",
            "_update_access_time",
            "_insert_file_metadata",
        }:
            # all the methods defined in this class. Note `open` here, since
            # it calls `_open`, but is actually in superclass
            return lambda *args, **kw: getattr(type(self), item).__get__(self)(*args, **kw)
        if item in ["__reduce_ex__"]:
            raise AttributeError
        if item in ["transaction"]:
            # property
            return type(self).transaction.__get__(self)
        if item in {
            "protocol",
        }:
            # class attributes
            return getattr(type(self), item)
        if item == "__class__":
            return type(self)
        d = object.__getattribute__(self, "__dict__")
        fs = d.get("fs", None)  # fs is not immediately defined
        if item in d:
            return d[item]
        elif fs is not None:
            if item in fs.__dict__:
                # attribute of instance
                return fs.__dict__[item]
            # attributed belonging to the target filesystem
            cls = type(fs)
            m = getattr(cls, item)
            if (inspect.isfunction(m) or inspect.isdatadescriptor(m)) and (
                not hasattr(m, "__self__") or m.__self__ is None
            ):
                # instance method
                return m.__get__(fs, cls)  # type: ignore
            return m  # class method or attribute
        else:
            # attributes of the superclass, while target is being set up
            return super().__getattribute__(item)

    def _check_file(self, path):
        path = self._strip_protocol(path)
        cache_key = _cache_mapper(path)
        fn = os.path.join(self.storage, cache_key)
        if os.path.exists(fn):
            return fn
        return False

    def _insert_file_metadata(self, path, fn):
        pass

    def _update_access_time(self, path, fn):
        # touch to update mtime
        Path(fn).touch()

    def _enforce_cache_size(self):
        if self.cache_storage_size is None:
            return

        current_time = time.time()
        if self._last_enforce_time + self.enforce_size_every_seconds > current_time:
            return

        def cached_files():
            """Return list of (filename, mtime) in cache, sorted by mtime ascending"""
            for entry in os.scandir(self.storage):
                if entry.is_file() and not entry.name.endswith(LOCK_FILE_SUFFIX):
                    try:
                        stat = entry.stat()
                        yield (entry.path, stat.st_mtime, stat.st_size)
                    except FileNotFoundError as e:
                        logger.debug("Could not stat file %s: %s", entry.path, e)

        try:
            with FileLock(f"{self.storage}/{CLEANUP_LOCK_FILE}", blocking=False, mode=LOCK_FILE_MODE):
                files = list(cached_files())
                cache_size = sum(size for _, _, size in files)

                if cache_size <= (self.cache_storage_size * self.cache_cleanup_high_watermark):
                    return

                files.sort(key=lambda x: x[1])

                difference = (cache_size * self.cache_cleanup_high_watermark) - (
                    self.cache_storage_size * self.cache_cleanup_high_watermark
                )
                logger.info(
                    "Cache size %d exceeds limit of %d, removing old files", cache_size, self.cache_storage_size
                )

                file_index = 0
                while difference > 0 and file_index < len(files):
                    to_remove, _, size = files[file_index]
                    file_index += 1
                    try:
                        # Acquire a lock before removing the file to avoid conflicts
                        with FileLock(to_remove + LOCK_FILE_SUFFIX, blocking=False, mode=LOCK_FILE_MODE):
                            os.remove(to_remove)
                        logger.debug("Removed cached file %s to free space", to_remove)
                        difference -= size
                    except Timeout:
                        logger.info("Could not acquire lock to remove cached file %s, skipping", to_remove)
                    except Exception as e:
                        logger.info("Could not remove cached file %s: %s", to_remove, e)
        except Timeout:
            logger.debug("Could not acquire cache cleanup lock, skipping cache size enforcement")
        self._last_enforce_time = current_time

    def _atomic_get(self, rpath: str, lpath: str, **kwargs):
        _, fn = tempfile.mkstemp(dir=os.path.dirname(lpath), prefix=os.path.basename(lpath) + "-")
        try:
            self.fs.get_file(rpath, fn, **kwargs)
            os.chmod(fn, self.cache_storage_mode)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(fn)
            raise
        else:
            os.replace(fn, lpath)

    def _open(self, path: str, mode="rb", **kwargs):
        if not isinstance(path, str):
            raise TypeError(f"Invalid path type: {type(path)}")

        path = self._strip_protocol(path)
        path = self.fs._strip_protocol(path)

        if "r" not in mode:
            raise NotImplementedError(f"Writes are not supported in {type(self)}")
        fn = self._check_file(path)
        if fn:
            f = open(fn, mode)
            self._update_access_time(path, fn)
            return f

        fn = os.path.join(self.storage, _cache_mapper(path))

        logger.debug("Copying %s to local cache", path)

        # Use a lock to protect file access during cache population
        with FileLock(fn + LOCK_FILE_SUFFIX, timeout=60, mode=LOCK_FILE_MODE):
            # Check if another process created the file while we were waiting
            if not os.path.exists(fn):
                self._atomic_get(path, fn)
            f = open(fn, mode)
        # Enforce cache size after releasing the lock
        self._enforce_cache_size()
        self._insert_file_metadata(path, fn)
        return f

    def cat_ranges(
        self,
        paths: list[str],
        starts: int | list[int],
        ends: int | list[int],
        max_gap=None,
        on_error="return",
        **kwargs,
    ):
        local_paths = [self._check_file(p) for p in paths]
        for p, l in zip(paths, local_paths, strict=False):
            if l:
                self._update_access_time(p, l)
        remote_paths = [p for l, p in zip(local_paths, paths, strict=False) if not l]
        local_paths = [
            os.path.join(self.storage, _cache_mapper(p)) for l, p in zip(local_paths, paths, strict=False) if not l
        ]
        if remote_paths:
            self._enforce_cache_size()
        with _MultiFileLock(local_paths, timeout=60):
            self.fs.get(remote_paths, local_paths)
            for l in local_paths:
                os.chmod(l, self.cache_storage_mode)
        for p, l in zip(remote_paths, local_paths, strict=False):
            self._insert_file_metadata(p, l)

        paths = [self._check_file(p) for p in paths]
        return self._local.cat_ranges(paths, starts, ends, max_gap=max_gap, on_error=on_error, **kwargs)


class SqliteLRUCacheFileSystem(LRUCacheFileSystem):
    """Caches whole remote files on first access, with SQLite metadata

    This class is intended as a layer over any other file system, and
    will make a local copy of each file accessed, so that all subsequent
    reads are local. This implementation only copies whole files, and
    keeps metadata about the download time and file details in a SQLite
    database. It is therefore safer to use in multi-threaded/concurrent
    situations.
    """

    protocol = "sqlitecache"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._db_path = os.path.join(self.storage, "cache_metadata.sqlite")
        self._ensure_schema()
        self._conn = sqlite3.connect(self._db_path, timeout=0, check_same_thread=False)
        if os.path.exists(self._db_path):
            os.chmod(self._db_path, self.cache_storage_mode)

    def _ensure_schema(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                path TEXT PRIMARY KEY,
                fn TEXT,
                time REAL,
                size INTEGER
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_metadata_time ON cache_metadata(time)")
        conn.close()

    @tenacity.retry(stop=tenacity.stop_after_attempt(10), wait=tenacity.wait_exponential_jitter(0.25, max=10))
    def _check_file(self, path) -> str | Literal[False]:
        path = self._strip_protocol(path)
        cursor = self._conn.cursor()
        cursor.execute("SELECT fn FROM cache_metadata WHERE path = ?", (path,))
        row = cursor.fetchone()
        cursor.close()
        if row:
            fn: str = row[0]
            if os.path.exists(fn):
                return fn
            else:
                # File missing, remove from DB
                with self._conn:
                    self._conn.execute("DELETE FROM cache_metadata WHERE path = ?", (path,))
        return False

    @tenacity.retry(stop=tenacity.stop_after_attempt(10), wait=tenacity.wait_exponential_jitter(0.25, max=10))
    def _update_access_time(self, path, fn):
        with self._conn:
            self._conn.execute("UPDATE cache_metadata SET time = ? WHERE path = ?", (time.time(), path))

    @tenacity.retry(stop=tenacity.stop_after_attempt(10), wait=tenacity.wait_exponential_jitter(0.25, max=10))
    def _insert_file_metadata(self, path, fn):
        size = os.path.getsize(fn)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache_metadata (path, fn, time, size) VALUES (?, ?, ?, ?)",
                (path, fn, time.time(), size),
            )

    def _enforce_cache_size(self):
        MAX_FILES_PER_CLEANUP = 32
        if self.cache_storage_size is None:
            return

        current_time = time.time()
        if self._last_enforce_time + self.enforce_size_every_seconds > current_time:
            return

        try:
            with FileLock(f"{self.storage}/{CLEANUP_LOCK_FILE}", blocking=False, mode=LOCK_FILE_MODE):
                cursor = self._conn.cursor()
                cursor.execute("SELECT SUM(size) FROM cache_metadata")
                cache_size = cursor.fetchone()[0] or 0
                cursor.close()
                if cache_size <= (self.cache_storage_size * self.cache_cleanup_high_watermark):
                    return

                difference = (cache_size * self.cache_cleanup_low_watermark) - (
                    self.cache_storage_size * self.cache_cleanup_high_watermark
                )

                cursor = self._conn.cursor()
                cursor.execute(
                    f"""
                SELECT fn, size
                FROM cache_metadata
                ORDER BY time ASC
                LIMIT {MAX_FILES_PER_CLEANUP}
                """,
                )
                # Fetch the results for processing
                files_to_delete = cursor.fetchall()
                logger.info(
                    "Cache size %d exceeds limit of %d, removing old files", cache_size, self.cache_storage_size
                )
                cursor.close()
                for fn, size in files_to_delete:
                    if difference <= 0:
                        break
                    try:
                        with FileLock(fn + LOCK_FILE_SUFFIX, blocking=False, mode=LOCK_FILE_MODE):
                            if os.path.exists(fn):
                                os.remove(fn)
                        with self._conn:
                            self._conn.execute("DELETE FROM cache_metadata WHERE fn = ?", (fn,))
                        logger.info("Removed cached file %s to free space", fn)
                        difference -= size
                    except Timeout:
                        logger.info("Could not acquire lock to remove cached file %s, skipping", fn)
                    except Exception as e:
                        logger.info("Could not remove cached file %s: %s", fn, e)

        except Timeout:
            logger.debug("Could not acquire cache cleanup lock, skipping cache size enforcement")
        self._last_enforce_time = current_time
