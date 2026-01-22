============
Disk Caching
============

The ``noether.io.diskcache.lru_cache`` module provides filesystem cache implementations that automatically manage local copies of remote files with Least Recently Used (LRU) eviction when size limits are exceeded.

What is fsspec?
===============

`fsspec <https://filesystem-spec.readthedocs.io/>`_ is a Python library that provides a unified interface for working with different filesystems. 
It allows you to interact with local files, cloud storage (S3, Azure, GCS), HTTP endpoints, and many other storage backends using the same API. 
The caching implementations in this module wrap any fsspec filesystem to add transparent local caching with automatic eviction.

Overview
========

When working with remote filesystems (S3, Azure Blob Storage, HTTP, etc.), repeatedly accessing the same files can be slow and costly. The LRU cache filesystems solve this by:

- **Transparently caching** files on first access
- **Automatically managing** cache size with configurable limits
- **Evicting least recently used** files when the cache grows too large
- **Thread-safe operation** with file locking to prevent race conditions
- **Supporting any fsspec filesystem** as the backing store

Available Implementations
=========================

:py:class:`noether.io.diskcache.lru_cache.LRUCacheFileSystem`
------------------------------------------------------------

A basic LRU cache implementation that uses filesystem metadata (modification times) to track file access patterns.
This is recommended in most scenarios, unless you have a high number of files you want to cache.

**Limitations:**

- Updates mtime on every file access (additional I/O overhead)
- May have performance issues on some networked filesystems
- Cache eviction has to scan the entire cache directory

:py:class:`noether.io.diskcache.lru_cache.SqliteLRUCacheFileSystem`
------------------------------------------------------------------

An enhanced implementation that uses SQLite to track cache metadata instead of relying on filesystem modification times.

**Advantages:**

- Optimized for high-volume cache storage
- More reliable metadata tracking

**Limitations:**

- Throughput is worse on fast local filesystems due to SQLite overhead

Basic Usage
===========

There are two main ways to work with fsspec filesystems: instantiating the filesystem directly or encoding the filesystem and its parameters in a URL.

Instantiate Directly
--------------------

.. code-block:: python

    import fsspec
    import s3fs # Required for S3 support
    from noether.io.diskcache.lru_cache import LRUCacheFileSystem

    s3 = s3fs.S3FileSystem(anon=False)  

    # Wrap any fsspec filesystem with caching
    fs = LRUCacheFileSystem(
        fs=s3,
        storage="/tmp/my_cache",
        cache_size=10**9,  # 1 GB limit
    )

    # First access downloads the file
    with fs.open("my-bucket/data.csv", "rb") as f:
        data = f.read()

    # Second access uses the cached copy
    with fs.open("my-bucket/data.csv", "rb") as f:
        data = f.read()  # Fast! No download needed

URL Syntax
----------
Note: Example requires ocifs package for OCI support.

.. code-block:: python

    import fsspec

    # First access downloads the file
    with fsspec.open("lrucache::oci://bucket@namespace/data.csv", "r", 
            lrucache={"storage": "/tmp/cache", "cache_size": 1024**2},
            oci={"config": "~/.oci/config"}
    ) as f:
        data = f.read()

    # Second access uses the cached copy
    with fsspec.open("lrucache::oci://bucket@namespace/data.csv", "r",
            lrucache={"storage": "/tmp/cache", "cache_size": 1024**2},
            oci={"config": "~/.oci/config"}) as f:
        data = f.read()  # Fast! No download needed

Advanced usage
==============

The cache filesystems support several configuration options to customize behavior:

Cleanup behaviour can be tuned with :py:attr:`cache_cleanup_high_watermark` and :py:attr:`cache_cleanup_low_watermark` to control when eviction is triggered and how much space to free.

Permissions can be set with :py:attr:`cache_storage_mode` to control access to cached files.

Example with Custom Configuration
----------------------------------

.. code-block:: python

    fs = LRUCacheFileSystem(
        fs=fsspec.filesystem("s3"),
        storage="/var/cache/s3_data",
        cache_size=50 * 10**9,  # 50 GB
        cache_storage_mode=0o644,  # More permissive file permissions
        enforce_size_every_seconds=5,  # Check less frequently
        cache_cleanup_high_watermark=0.9,  # Trigger at 90% full
        cache_cleanup_low_watermark=0.7,  # Clean down to 70%
    )

Cache Management
================

Automatic Eviction
------------------

When the cache size exceeds ``cache_size * cache_cleanup_high_watermark``, the filesystem automatically:

1. Acquires a cleanup lock to prevent conflicts
2. Identifies the least recently used files
3. Removes files until cache size reaches ``cache_size * cache_cleanup_low_watermark``
4. Skips files that are currently locked (in use)

Wrapping Different Filesystems
------------------------------

The cache works with any fsspec-compatible filesystem:

.. code-block:: python

    # Oracle Cloud Infrastructure (OCI) Object Storage
    oci_fs = LRUCacheFileSystem(
        fs=fsspec.filesystem("oci", config="~/.oci/config"),
        storage="/tmp/oci_cache",
        cache_size=10**9,
    )

    # HTTP/HTTPS
    http_fs = LRUCacheFileSystem(
        fs=fsspec.filesystem("http"),
        storage="/tmp/http_cache",
        cache_size=500 * 10**6,
    )

    # Google Cloud Storage
    gcs_fs = SqliteLRUCacheFileSystem(
        fs=fsspec.filesystem("gcs"),
        storage="/tmp/gcs_cache",
        cache_size=20 * 10**9,
    )

Performance Considerations
==========================

Choosing the Right Implementation
---------------------------------

Use **LRUCacheFileSystem** when:

- Running single-threaded applications
- Filesystem supports fast mtime updates
- Simplicity is preferred over maximum performance

Use **SqliteLRUCacheFileSystem** when:

- Running multi-threaded or concurrent applications
- Working with high-frequency file access patterns
- Maximum performance is required
- Cache directory is on a networked filesystem

Limitations
===========

- **Write operations are not supported**: The cache is read-only. Attempts to open files in write mode will raise ``NotImplementedError``
- **Whole file caching only**: Files are cached in their entirety, not incrementally
- **Local storage required**: Cache requires local disk space
- **No distributed cache**: Each process/machine maintains its own independent cache