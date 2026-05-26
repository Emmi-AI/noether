#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import boto3  # type: ignore[import-untyped]
from boto3.s3.transfer import TransferConfig  # type: ignore[import-untyped]
from botocore import UNSIGNED  # type: ignore[import-untyped]
from botocore.client import BaseClient  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]
from loguru import logger

from noether.io.credentials import get_credentials
from noether.io.providers import Provider


class AWSSecrets(str, Enum):
    AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
    AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
    AWS_SESSION_TOKEN = "AWS_SESSION_TOKEN"
    AWS_REGION = "AWS_REGION"
    AWS_DEFAULT_REGION = "AWS_DEFAULT_REGION"
    AWS_ENDPOINT_URL = "AWS_ENDPOINT_URL"


class S3Object(TypedDict):
    key: str
    size: int
    etag: str | None


@lru_cache(maxsize=1)
def get_s3_client() -> BaseClient:
    """
    Construct an S3 client from managed credentials (env or config).
    Expected keys (matching env names):
      - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
      - optional AWS_SESSION_TOKEN
      - AWS_DEFAULT_REGION or AWS_REGION
      - optional AWS_ENDPOINT_URL (for MinIO / on‑prem / custom endpoints)

    Falls back to unsigned access for public buckets when no non‑empty
    credentials are provided.
    """
    credentials = get_credentials(Provider.AWS)

    def _clean(value: str | None) -> str | None:
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    # Prefer AWS_DEFAULT_REGION, but allow AWS_REGION as a fallback:
    region = _clean(credentials.get(AWSSecrets.AWS_DEFAULT_REGION, None)) or _clean(
        credentials.get(AWSSecrets.AWS_REGION, None)
    )

    # Optional custom endpoint (e.g., MinIO). We don't require it to be in the enum for backwards compatibility:
    endpoint_url = _clean(credentials.get(getattr(AWSSecrets, "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL"), None))

    kwargs: dict[str, str] = {}
    akid = _clean(credentials.get(AWSSecrets.AWS_ACCESS_KEY_ID, None))
    secret = _clean(credentials.get(AWSSecrets.AWS_SECRET_ACCESS_KEY, None))
    token = _clean(credentials.get(AWSSecrets.AWS_SESSION_TOKEN, None))

    if akid:
        kwargs["aws_access_key_id"] = akid
    if secret:
        kwargs["aws_secret_access_key"] = secret
    if token:
        kwargs["aws_session_token"] = token
    if region:
        kwargs["region_name"] = region

    # Shared client configuration: retries and connection pool tuning:
    cfg = Config(
        retries={"max_attempts": 10, "mode": "standard"},
        max_pool_connections=64,
    )

    if not kwargs:
        logger.info("Using unsigned S3 client (public bucket)")
        return boto3.client("s3", config=Config(signature_version=UNSIGNED, **cfg.__dict__["_user_provided_options"]))

    logger.info("Using credentialed S3 client")
    # Pass endpoint_url only if provided (helps with MinIO / on‑prem):
    if endpoint_url:
        return boto3.client("s3", endpoint_url=endpoint_url, config=cfg, **kwargs)
    return boto3.client("s3", config=cfg, **kwargs)


def _is_dir_marker(key: str, keyset: set[str]) -> bool:
    """
    Heuristic: treat a zero-byte 'directory marker' object as a directory placeholder
    if any other key starts with 'key + /'.
    """
    # Normalize: markers we see are often like 'foo/' and are already skipped elsewhere,
    # but on some systems zero-byte 'foo' can appear. Consider it a marker if subkeys exist.
    return any(k.startswith(key.rstrip("/") + "/") for k in keyset if k != key)


def _preflight_fix_directory_conflicts(local_root: Path, keys: list[str]) -> None:
    """
    Ensure that future directory creates will not fail due to a pre-existing *file*
    where a directory needs to exist. If such a zero-byte file exists, remove it.
    """
    # Build the set of directory paths that will be needed
    dir_paths: set[Path] = set()
    for key in keys:
        p = local_root / key
        dir_paths.add(p.parent)

    for d in sorted(dir_paths):
        if d.exists() and d.is_file():
            # Remove conflicting zero-byte file so we can create a directory there
            try:
                if d.stat().st_size == 0:
                    d.unlink()
            except Exception as exc:
                logger.warning(f"Could not remove conflicting file at {d}: {exc}")
        d.mkdir(parents=True, exist_ok=True)


def list_s3_objects(
    bucket: str,
    prefix: str,
    extension: str | None = None,
) -> list[S3Object]:
    """
    List S3 objects under bucket/prefix with an optional extension filter. Skips directory placeholders (keys ending with '/') and normalizes quoted ETags.
    """
    client = get_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    page_it = paginator.paginate(Bucket=bucket, Prefix=prefix)

    results: list[S3Object] = []
    for page in page_it:
        contents = page.get("Contents") or []
        for item in contents:
            key = item.get("Key")
            if not key or key.endswith("/"):
                # skip folder markers
                continue
            if extension and not key.endswith(extension):
                continue
            size = int(item.get("Size", 0))
            etag = item.get("ETag")
            if isinstance(etag, str):
                etag = etag.strip('"')
            results.append(S3Object(key=key, size=size, etag=etag))
    return results


def estimate_s3_size(
    bucket: str,
    prefix: str,
    extension: str | None = None,
) -> tuple[int, int]:
    """Estimate size of objects under bucket/prefix with an optional extension filter.

    Args:
        bucket: Name of the S3 bucket.
        prefix: File prefix.
        extension: Optional file extension. Defaults to None.

    Returns:
        - A tuple with estimated size in bytes and total number of objects.
    """
    objects = list_s3_objects(bucket, prefix, extension=extension)
    total = sum(x["size"] for x in objects)
    return total, len(objects)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fetch_s3_file(
    bucket: str,
    key: str,
    local_dir: Path,
) -> Path:
    """Download file from S3 bucket to local directory, preserving the key's subpath.

    Args:
        bucket: Name of the S3 bucket.
        key: File key.
        local_dir: Path to local directory.

    Returns:
        - Local file path.
    """
    client = get_s3_client()
    local_path = local_dir / key
    _ensure_parent(local_path)
    transfer_cfg = TransferConfig(max_concurrency=8, multipart_threshold=8 * 1024 * 1024)
    client.download_file(bucket, key, str(local_path), Config=transfer_cfg)
    return local_path


def iter_s3_object_chunks(bucket: str, key: str, *, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    """Stream an S3 object as chunks of bytes. Intended to be used with higher-level atomic writers / hashing
    in the CLI.

    Args:
        bucket: S3 bucket name.
        key: S3 object key.
        chunk_size: Size of chunks in bytes.

    Yields:
        Byte chunks from the object body.
    """
    client = get_s3_client()
    resp = client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"]
    # botocore.response.StreamingBody exposes iter_chunks:
    for chunk in body.iter_chunks(chunk_size=chunk_size):
        if chunk:
            yield chunk


def head_s3_object(bucket: str, key: str) -> tuple[int | None, str | None]:
    """Lightweight HEAD to retrieve content length and etag (if available).
    Returns:
        (size_bytes, etag) with etag normalized (quotes stripped).
    """
    client = get_s3_client()
    try:
        response = client.head_object(Bucket=bucket, Key=key)
    except Exception:
        return None, None
    size = int(response.get("ContentLength", 0)) if "ContentLength" in response else None
    etag = response.get("ETag")
    if isinstance(etag, str):
        etag = etag.strip('"')
    return size, etag


def fetch_s3_prefix(
    bucket: str,
    prefix: str,
    local_dir: Path,
    extension: str | None = None,
    max_workers: int = 8,
) -> list[str]:
    """Download all objects under bucket/prefix with an optional extension filter into a local directory.

    Args:
        bucket: Name of the S3 bucket.
        prefix: File prefix.
        local_dir: Path to local directory.
        extension: Optional file extension. Defaults to None.
        max_workers: Number of workers to use for downloading. Defaults to 8.

    Returns:
        - A list of relative paths (keys) written.
    """
    # First list, then download in a thread pool; order of results follows completion, not lexicographic.
    objects = list_s3_objects(bucket, prefix, extension=extension)
    if not objects:
        logger.warning(f"No S3 objects found under s3://{bucket}/{prefix} (extension={extension!r})")
        return []

    logger.info(f"Downloading {len(objects)} S3 objects from s3://{bucket}/{prefix} ...")
    # Ensure local directory tree won't conflict with zero-byte files:
    _preflight_fix_directory_conflicts(local_dir, [o["key"] for o in objects])

    local_dir.mkdir(parents=True, exist_ok=True)

    relative_paths: list[str] = []
    failures: list[tuple[str, BaseException]] = []

    def _dl(obj: S3Object) -> str:
        path = fetch_s3_file(bucket, obj["key"], local_dir)
        return str(path.relative_to(local_dir).as_posix())

    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(objects)))) as executor:
        # Map each Future to the corresponding S3 key for tracking:
        future_to_s3_key: dict[Future[str], str] = {executor.submit(_dl, obj): obj["key"] for obj in objects}

        for completed_future in as_completed(future_to_s3_key):
            s3_key = future_to_s3_key[completed_future]
            try:
                downloaded_relative_path = completed_future.result()
                relative_paths.append(downloaded_relative_path)
            except BaseException as download_error:
                failures.append((s3_key, download_error))

    if failures:
        for key, exc in failures:
            logger.error(f"Failed to download s3://{bucket}/{key} → {exc}")
        # raise RuntimeError(f"{len(failures)} S3 downloads failed")

    logger.info(f"Download complete → {local_dir}")
    return relative_paths
