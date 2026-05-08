#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from hydra.core.global_hydra import GlobalHydra

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RECIPES_DIR = _REPO_ROOT / "recipes"

# Top-level package names exposed by recipes. When a recipe directory goes onto sys.path these names take over;
# we flush them on swap so a previous recipe's cached module does not shadow the new one.
_AERO_CFD_TOP_LEVEL = ("pipeline", "trainers", "callbacks", "model", "showcase")
_HEAT_TRANSFER_TOP_LEVEL = ("pipeline", "callbacks", "model")


def _ensure_path(path: Path) -> None:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_path(_RECIPES_DIR)


@pytest.fixture
def device() -> str:
    """Return ``"cuda"`` if available, otherwise ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def accelerator(device: str) -> str:
    """The noether accelerator string (``"gpu"`` or ``"cpu"``)."""
    return "gpu" if device == "cuda" else "cpu"


@pytest.fixture(autouse=True)
def _clear_hydra() -> Iterator[None]:
    """Clear Hydra's global state before and after each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _put_recipe_on_path(monkeypatch: pytest.MonkeyPatch, recipe_dir: Path, top_level_names: tuple[str, ...]) -> None:
    monkeypatch.syspath_prepend(str(recipe_dir))
    # Drop cached top-level modules from any previously activated recipe so the next ``import pipeline`` (etc.)
    # re-resolves against the new sys.path.
    for name in top_level_names:
        sys.modules.pop(name, None)


@pytest.fixture
def aero_cfd_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Activate ``recipes/aero_cfd/`` for top-level imports used in YAML kinds."""
    _put_recipe_on_path(monkeypatch, _RECIPES_DIR / "aero_cfd", _AERO_CFD_TOP_LEVEL)


@pytest.fixture
def heat_transfer_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Activate ``recipes/heat_transfer/`` for top-level imports used in YAML kinds."""
    _put_recipe_on_path(monkeypatch, _RECIPES_DIR / "heat_transfer", _HEAT_TRANSFER_TOP_LEVEL)
