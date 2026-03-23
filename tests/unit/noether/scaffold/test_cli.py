#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from noether.scaffold.cli import app

runner = CliRunner()


@pytest.mark.parametrize("bad_name", ["123bad", "with-hyphen", "has space"], ids=["leading-digit", "hyphen", "space"])
def test_invalid_project_name_rejected(tmp_path: Path, bad_name: str) -> None:
    result = runner.invoke(
        app,
        [
            bad_name,
            "--project-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "not a valid Python identifier" in result.output


def test_existing_directory_rejected(tmp_path: Path) -> None:
    project_dir = tmp_path / "existing_proj"
    project_dir.mkdir()
    result = runner.invoke(
        app,
        [
            "existing_proj",
            "--project-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "Directory already exists" in result.output


def test_valid_invocation_succeeds(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "my_project",
            "--project-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    proj = tmp_path / "my_project"
    pkg = proj / "my_project"  # nested package folder with same name as project
    assert proj.is_dir()
    assert (proj / "pyproject.toml").is_file()
    assert (pkg / "callbacks").is_dir()
    assert (pkg / "configs").is_dir()
    assert (pkg / "models").is_dir()
    assert (pkg / "pipelines").is_dir()
    assert (pkg / "schemas").is_dir()
    assert (pkg / "trainer").is_dir()
    assert (pkg / "datasets").is_dir()
