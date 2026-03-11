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
            "--model",
            "upt",
            "--dataset",
            "shapenet_car",
            "--dataset-path",
            "/tmp/x",
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
            "--model",
            "upt",
            "--dataset",
            "shapenet_car",
            "--dataset-path",
            "/tmp/x",
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
            "--model",
            "upt",
            "--dataset",
            "shapenet_car",
            "--dataset-path",
            "/tmp/fake_data",
            "--project-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "my_project").is_dir()
    assert (tmp_path / "my_project" / "callbacks").is_dir()
    assert (tmp_path / "my_project" / "configs").is_dir()
    assert (tmp_path / "my_project" / "model").is_dir()
    assert (tmp_path / "my_project" / "pipeline").is_dir()
    assert (tmp_path / "my_project" / "schemas").is_dir()
    assert (tmp_path / "my_project" / "trainers").is_dir()
