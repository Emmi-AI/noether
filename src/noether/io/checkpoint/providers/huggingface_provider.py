#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests  # type: ignore[import-untyped]
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers

from noether.io.checkpoint.iohash import DEFAULT_CHUNK, atomic_write_and_hash
from noether.io.checkpoint.types import Provider
from noether.io.credentials import get_credentials
from noether.io.logging_progress import LogProgress
from noether.io.providers import Provider as CredentialsProvider


class HFProvider(Provider):
    """Provides a minimalistic interface to fetch a checkpoint file from Huggingface.

    Examples:
        .. code-block:: console

            hf://org/name@rev?filename=model.pt
            hf://org/name?filename=model.pt

        Where:
          - ``rev`` is optional (tag/branch/commit after ``@``)
          - ``filename`` is required (query parameter ``?filename=...``)
    """

    scheme = "hf"

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("hf://")

    @staticmethod
    def parse_uri(uri: str) -> tuple[str, str, str | None]:
        """Extracts repo name, revision and filename from HF URI.

        Args:
            uri: Input URI as: 'hf://org/name?filename=MODEL_FILENAME'.

        Returns:
            - tuple[str, str, Optional[str]]: Repository name, filename and revision.
        """
        parsed = urlparse(uri)
        if parsed.scheme != HFProvider.scheme:
            return "", "", None
        path = parsed.netloc + parsed.path  # 'org/name' (netloc='org', path='/name')
        if "@" in path:
            repo_id, revision = path.split("@", 1)
        else:
            repo_id, revision = path, "main"
        qs = parse_qs(parsed.query)
        filename = (qs.get("filename") or [None])[0]  # type: ignore[list-item]
        if not repo_id or not filename:
            raise ValueError("hf:// must be 'hf://org/name?filename=MODEL_FILENAME'")
        return repo_id, filename, revision

    def estimate_size(self, uri: str) -> int | None:
        """Extracts an approximate size of a file from HF URI.

        Args:
            uri: Input URI as: 'hf://org/name[@rev]?filename=MODEL_FILENAME'.

        Returns:
            - int: Estimated size of the remote file.
            - None: If no size is available.
        """
        try:
            repo, filename, revision = self.parse_uri(uri)
            token = get_credentials(CredentialsProvider.HUGGINGFACE).get("HF_TOKEN")
            api = HfApi()
            # Use model_info/dataset_info heuristics: first try model, then dataset:
            for fn in (api.model_info, api.dataset_info):
                try:
                    info = fn(repo, revision=revision, token=token)
                    for sib in getattr(info, "siblings", []) or []:
                        if getattr(sib, "rfilename", None) == filename and getattr(sib, "size", None) is not None:
                            return int(sib.size)
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def fetch(self, uri: str, destination_dir: Path, *, compute_hash: bool = True) -> tuple[Path, str | None]:
        """Returns a local file path for the given HF URI after downloading from the HF hub.

        Args:
            uri: Input URI where the file is located.
            destination_dir: Destination directory where the file need to be located.
            compute_hash: Compute the hash of the file on the local file system.

        Returns:
            - Path: Local file path.
        """
        repo_id, filename, revision = self.parse_uri(uri)
        token = get_credentials(CredentialsProvider.HUGGINGFACE).get("HF_TOKEN")
        url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)

        final_path = destination_dir / filename
        headers = build_hf_headers(token=token)

        with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length") or 0)
            progress = LogProgress(label=final_path.name, total_bytes=total)
            sha = atomic_write_and_hash(
                final_path,
                resp.iter_content(DEFAULT_CHUNK),
                compute_hash=compute_hash,
                progress=progress,
            )
        return final_path, sha
