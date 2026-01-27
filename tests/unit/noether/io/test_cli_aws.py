#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import io
import json
from pathlib import Path

from typer.testing import CliRunner

from noether.io.cli import cli_aws

runner = CliRunner(mix_stderr=False)

MODULE_PATH = "noether.io.cli.cli_aws"


class _FakeS3Client:
    def __init__(self, data_by_key: dict[str, bytes] | None = None, fail_keys: set[str] | None = None):
        self.data_by_key = data_by_key or {}
        self.fail_keys = fail_keys or set()

    def download_fileobj(self, *, Bucket: str, Key: str, Fileobj: io.BufferedWriter):
        if Key in self.fail_keys:
            raise RuntimeError("forced download error")
        Fileobj.write(self.data_by_key.get(Key, b""))


def test_estimate_happy_path(monkeypatch):
    monkeypatch.setattr(
        f"{MODULE_PATH}.estimate_s3_size",
        lambda bucket, prefix, extension=None: (123456, 7),
    )
    res = runner.invoke(
        cli_aws.aws_s3_app,
        ["estimate", "buck", "pref/"],
    )
    assert res.exit_code == 0
    assert "→ 7 files" in res.stdout
    assert "120.56 KB" in res.stdout or "120.6 KB" in res.stdout  # fmt_bytes rounding tolerance


def test_file_download_prints_written_path(monkeypatch, tmp_path: Path):
    def _fake_fetch(bucket: str, key: str, local_dir: Path) -> Path:
        path = local_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")
        return path

    monkeypatch.setattr(f"{MODULE_PATH}.fetch_s3_file", _fake_fetch)

    res = runner.invoke(
        cli_aws.aws_s3_app,
        ["file", "buck", "x/a.bin", str(tmp_path)],
    )
    assert res.exit_code == 0
    assert "Wrote:" in res.stdout
    assert (tmp_path / "x" / "a.bin").read_bytes() == b"ok"


def test_fetch_dry_run_reports_counts(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        f"{MODULE_PATH}.list_s3_objects",
        lambda bucket, prefix, extension=None: [
            {"key": "p/a.bin", "size": 2},
            {"key": "p/b.bin", "size": 3},
        ],
    )
    res = runner.invoke(
        cli_aws.aws_s3_app,
        ["fetch", "buck", "p/", str(tmp_path), "--dry-run"],
    )
    assert res.exit_code == 0
    assert "[dry-run]" in res.stdout
    assert not any(tmp_path.rglob("*"))  # nothing created


def test_fetch_empty_writes_empty_manifest_if_requested(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        f"{MODULE_PATH}.list_s3_objects",
        lambda *a, **kw: [],
    )
    manifest_path = tmp_path / "m.json"
    res = runner.invoke(
        cli_aws.aws_s3_app,
        ["fetch", "buck", "p/", str(tmp_path), "--manifest-out", str(manifest_path)],
    )
    assert res.exit_code == 0
    # manifest file created (empty dict):
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data == {}


# --------- fetch: happy path basic ---------
def test_fetch_streams_and_saves(monkeypatch, tmp_path: Path):
    # directory marker + two real files
    objects = [
        {"key": "p/", "size": 0},
        {"key": "p/a.txt", "size": 1},
        {"key": "p/b.txt", "size": 1},
    ]
    data = {"p/a.txt": b"A", "p/b.txt": b"B"}
    fake_client = _FakeS3Client(data)

    monkeypatch.setattr(f"{MODULE_PATH}.list_s3_objects", lambda *a, **kw: objects)
    monkeypatch.setattr(f"{MODULE_PATH}.get_s3_client", lambda: fake_client)

    res = runner.invoke(
        cli_aws.aws_s3_app,
        ["fetch", "buck", "p/", str(tmp_path)],
    )
    assert res.exit_code == 0, res.output
    assert (tmp_path / "p" / "a.txt").read_bytes() == b"A"
    assert (tmp_path / "p" / "b.txt").read_bytes() == b"B"
    assert "Download complete" in res.stdout


def test_fetch_strip_prefix(monkeypatch, tmp_path: Path):
    objects = [
        {"key": "pref/x/one.bin", "size": 1},
        {"key": "pref/x/two.bin", "size": 1},
    ]
    data = {o["key"]: b"1" for o in objects}
    fake_client = _FakeS3Client(data)

    monkeypatch.setattr(f"{MODULE_PATH}.list_s3_objects", lambda *a, **kw: objects)
    monkeypatch.setattr(f"{MODULE_PATH}.get_s3_client", lambda: fake_client)

    res = runner.invoke(
        cli_aws.aws_s3_app,
        ["fetch", "buck", "pref/", str(tmp_path), "--strip-prefix"],
    )
    assert res.exit_code == 0
    # Saved relative to strip prefix => paths start at x/...:
    assert (tmp_path / "x" / "one.bin").exists()
    assert (tmp_path / "x" / "two.bin").exists()


def test_fetch_flatten_no_collisions(monkeypatch, tmp_path: Path):
    objects = [
        {"key": "a/one.bin", "size": 1},
        {"key": "b/two.bin", "size": 1},
    ]
    data = {o["key"]: b"1" for o in objects}
    fake_client = _FakeS3Client(data)

    monkeypatch.setattr(f"{MODULE_PATH}.list_s3_objects", lambda *a, **kw: objects)
    monkeypatch.setattr(f"{MODULE_PATH}.get_s3_client", lambda: fake_client)

    res = runner.invoke(
        cli_aws.aws_s3_app,
        ["fetch", "buck", "base/", str(tmp_path), "--flatten"],
    )
    assert res.exit_code == 0
    # Only basenames used:
    assert (tmp_path / "one.bin").exists()
    assert (tmp_path / "two.bin").exists()


def test_fetch_flatten_collision_errors(monkeypatch, tmp_path: Path):
    objects = [
        {"key": "a/same.bin", "size": 1},
        {"key": "b/same.bin", "size": 1},
    ]
    fake_client = _FakeS3Client({})

    monkeypatch.setattr(f"{MODULE_PATH}.list_s3_objects", lambda *a, **kw: objects)
    monkeypatch.setattr(f"{MODULE_PATH}.get_s3_client", lambda: fake_client)

    res = runner.invoke(
        cli_aws.aws_s3_app,
        ["fetch", "buck", "base/", str(tmp_path), "--flatten"],
    )
    # Typer.BadParameter -> exit code != 0:
    assert res.exit_code != 0
    assert "Basename collisions" in res.stdout or "Basename collisions" in res.stderr


# --------- fetch: manifest + hashing ---------
def test_fetch_manifest_with_hash(monkeypatch, tmp_path: Path):
    objects = [
        {"key": "p/a.bin", "size": 1},
        {"key": "p/b.bin", "size": 2},
    ]
    data = {"p/a.bin": b"A", "p/b.bin": b"BC"}
    fake_client = _FakeS3Client(data)

    monkeypatch.setattr(f"{MODULE_PATH}.list_s3_objects", lambda *a, **kw: objects)
    monkeypatch.setattr(f"{MODULE_PATH}.get_s3_client", lambda: fake_client)
    monkeypatch.setattr(
        f"{MODULE_PATH}.hash_file",
        lambda p, chunk_size_mb=8: "nobody" if Path(p).name == "a.bin" else "readsthis",
    )

    mf = tmp_path / "manifest.json"
    res = runner.invoke(
        cli_aws.aws_s3_app,
        ["fetch", "buck", "p/", str(tmp_path), "--manifest-out", str(mf), "--hash"],
    )
    assert res.exit_code == 0
    data = json.loads(mf.read_text())
    assert "p/a.bin" in data
    assert data["p/a.bin"]["hash"] == "nobody"
    assert data["p/b.bin"]["hash"] == "readsthis"


def test_fetch_manifest_to_stdout(monkeypatch, tmp_path: Path):
    objects = [{"key": "p/a.bin", "size": 1}]
    fake_client = _FakeS3Client({"p/a.bin": b"A"})

    monkeypatch.setattr(f"{MODULE_PATH}.list_s3_objects", lambda *a, **kw: objects)
    monkeypatch.setattr(f"{MODULE_PATH}.get_s3_client", lambda: fake_client)
    monkeypatch.setattr(f"{MODULE_PATH}.hash_file", lambda *a, **kw: "h")

    res = runner.invoke(
        cli_aws.aws_s3_app,
        ["fetch", "buck", "p/", str(tmp_path), "--manifest-out", "-", "--hash"],
    )
    assert res.exit_code == 0
    # JSON echo to stdout:
    js = json.loads(res.stdout)
    assert list(js.keys()) == ["p/a.bin"]
    assert js["p/a.bin"]["hash"] == "h"


def test_fetch_download_failure_causes_exit_1(monkeypatch, tmp_path: Path):
    objects = [
        {"key": "p/a.bin", "size": 1},
        {"key": "p/b.bin", "size": 1},
    ]
    fake_client = _FakeS3Client({"p/a.bin": b"A"}, fail_keys={"p/b.bin"})

    monkeypatch.setattr(f"{MODULE_PATH}.list_s3_objects", lambda *a, **kw: objects)
    monkeypatch.setattr(f"{MODULE_PATH}.get_s3_client", lambda: fake_client)

    res = runner.invoke(
        cli_aws.aws_s3_app,
        ["fetch", "buck", "p/", str(tmp_path)],
    )
    assert res.exit_code != 0
    assert "[download-fail]" in (res.stdout + res.stderr)


def test_fetch_strict_hash_exits_on_hash_failure(monkeypatch, tmp_path: Path):
    objects = [
        {"key": "p/a.bin", "size": 1},
        {"key": "p/b.bin", "size": 1},
    ]
    fake_client = _FakeS3Client({"p/a.bin": b"A", "p/b.bin": b"B"})

    monkeypatch.setattr(f"{MODULE_PATH}.list_s3_objects", lambda *a, **kw: objects)
    monkeypatch.setattr(f"{MODULE_PATH}.get_s3_client", lambda: fake_client)

    def _hash(path: Path, chunk_size_mb: int = 8) -> str:
        # Fail on b.bin:
        if Path(path).name == "b.bin":
            raise RuntimeError("hash fail")
        return "ok"

    monkeypatch.setattr(f"{MODULE_PATH}.hash_file", _hash)

    res = runner.invoke(
        cli_aws.aws_s3_app,
        ["fetch", "buck", "p/", str(tmp_path), "--manifest-out", str(tmp_path / "m.json"), "--hash", "--strict-hash"],
    )
    assert res.exit_code != 0
    assert "[hash-fail]" in (res.stdout + res.stderr)
