#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import requests  # type: ignore[import-untyped]
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, snapshot_download
from loguru import logger

from noether.io.credentials import get_credentials
from noether.io.providers import Provider

HFRepoType = Literal["model", "dataset"]


def _get_hf_token() -> str | None:
    """Retrieve the HuggingFace token from credentials, or return None if unavailable.

    If credentials cannot be retrieved, the failure is logged at INFO level and
    None is returned so callers can proceed without authentication.

    Returns:
        The HF_TOKEN string, or None if credentials are not configured.
    """
    try:
        credentials = get_credentials(Provider.HUGGINGFACE)
        return str(credentials["HF_TOKEN"])
    except Exception as e:
        logger.info("Failed to get HuggingFace credentials, proceeding without authentication: %s", e)
        return None


def estimate_hf_repo_size(
    repo_id: str,
    repo_type: HFRepoType = "model",
    revision: str = "main",
    extension: str | None = None,
) -> int:
    """
    Estimate total size (bytes) of all files in a HF repo (model or dataset),
    optionally filtering by file-extension.

    Args:
        repo_id: HF repo ID, e.g. "bert-base-uncased" or "user/my-dataset"
        repo_type: "model" or "dataset"
        revision: branch/tag (default "main")
        extension: if given (e.g. ".jsonl"), only count files ending with this

    Returns:
        - Integer value for the total size in bytes.
    """
    creds = get_credentials(Provider.HUGGINGFACE)
    api = HfApi()

    if repo_type == "model":
        info = api.model_info(
            repo_id=repo_id,
            revision=revision,
            token=creds["HF_TOKEN"],
        )
    elif repo_type == "dataset":
        info = api.dataset_info(
            repo_id=repo_id,
            revision=revision,
            token=creds["HF_TOKEN"],
        )  # type: ignore[assignment]
    else:
        raise ValueError(
            f"Failed to determine repo size {repo_id!r} for {repo_type!r}. Supported types: 'model', 'dataset'"
        )

    # total = sum(file.size for file in info.siblings if extension is None or file.rfilename.endswith(extension))
    # logger.debug(f"Estimated repo size {total/1e6:.1f} MB")

    total = 0
    headers = {"Authorization": f"Bearer {creds['HF_TOKEN']}"}

    for sib in info.siblings:  # type: ignore[union-attr]
        if extension and not sib.rfilename.endswith(extension):
            continue

        if sib.size is not None:
            total += sib.size
        else:
            # Fallback for LFS or missing metadata
            url = hf_hub_url(repo_id, sib.rfilename, revision=revision)
            try:
                resp = requests.head(url, headers=headers, allow_redirects=True, timeout=5)
                resp.raise_for_status()
                content_length = resp.headers.get("Content-Length")
                if content_length:
                    total += int(content_length)
            except Exception:
                pass  # Ignore failures for inaccessible files

    return total


def fetch_huggingface_repo_snapshot(repo_id: str, local_dir: Path) -> None:
    """Downloads all content from the specific HuggingFace repository.

    Args:
        repo_id: ID of the HuggingFace repository.
        local_dir: Local directory to download content to.

    Returns:
        - None
    """
    credentials = get_credentials(Provider.HUGGINGFACE)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), token=credentials["HF_TOKEN"])


def fetch_huggingface_file(
    repo_id: str,
    filename: str,
    local_dir: Path,
    repo_type: HFRepoType = "model",
    revision: str = "main",
) -> None:
    """Downloads a specific file from a HuggingFace repository into a local directory.

    Args:
        repo_id: ID of the HuggingFace repository.
        filename: Filename to download.
        local_dir: Local directory to download the file to.
        repo_type: Repo type, either "model" or "dataset". Defaults to "model".
        revision: Revision of the repository. Defaults to "main".

    Returns:
        - None
    """
    token = _get_hf_token()
    local_dir.mkdir(parents=True, exist_ok=True)

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        token=token,
        local_dir=str(local_dir),
    )


def fetch_huggingface_by_extension(
    repo_id: str,
    extension: str,
    local_dir: Path,
    revision: str = "main",
    repo_type: HFRepoType = "dataset",
    max_workers: int = 8,
) -> list[str]:
    """Downloads specific files from a HuggingFace repository with given extension.

    Args:
        repo_id: ID of the HuggingFace repository.
        extension: File extension to download.
        local_dir: Local directory to download the file to.
        revision: Revision of the repository. Defaults to "main".
        repo_type: Repo type, either "model" or "dataset". Defaults to "dataset".
        max_workers: Maximum number of workers to use for downloading.

    Returns:
        - A list of downloaded files.
    """
    token = _get_hf_token()
    local_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        revision=revision,
    )

    # Filter only files with given extension:
    filtered_files = [f for f in files if f.endswith(extension)]
    if not filtered_files:
        logger.warning(f"No files with extension '{extension}' found in {repo_id}")
        return []

    logger.info(f"Downloading {len(filtered_files)} files with extension '{extension}' from {repo_id}...")

    downloaded_files: list[str] = []

    def _download(fname: str) -> None:
        hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type=repo_type,
            revision=revision,
            token=token,
            local_dir=str(local_dir),
        )
        downloaded_files.append(fname)

    # Currently, the stdout will be updated every time one of the downloads is finished - this will look a bit ugly
    # but unfortunately HuggingFace API doesn't expose options to alter/edit looks.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # for res in executor.map(_download, filtered_files):
        #     pass  # will raise if any _download fails
        list(executor.map(_download, filtered_files))

    logger.info(f"Download complete → {local_dir}")
    return downloaded_files
