#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from noether.training.cli.submit_job import (
    SlurmConfig,
    _expand_sweeps,
    _format_task_preview,
    _parse_argv,
    _train_entrypoint,
    _validate_all_combos,
    main,
    validate_config,
)

_MODULE_PATH = "noether.training.cli.submit_job"


# --------------------------------------------------------------------------- #
# _parse_argv
# --------------------------------------------------------------------------- #
class TestParseArgv:
    def test_positional_yaml_is_config(self):
        path, ov, mr, dr = _parse_argv(["prog", "cfg.yaml"])
        assert path == "cfg.yaml"
        assert ov == []
        assert not mr and not dr

    def test_hp_flag_is_config(self):
        path, ov, mr, dr = _parse_argv(["prog", "--hp", "cfg.yaml"])
        assert path == "cfg.yaml"

    def test_collects_overrides(self):
        path, ov, mr, dr = _parse_argv(["prog", "--hp", "cfg.yaml", "a=1", "b.c=2"])
        assert ov == ["a=1", "b.c=2"]

    def test_multirun_short_and_long(self):
        _, _, mr1, _ = _parse_argv(["prog", "--hp", "cfg.yaml", "-m"])
        _, _, mr2, _ = _parse_argv(["prog", "--hp", "cfg.yaml", "--multirun"])
        assert mr1 and mr2

    def test_dry_run_flag(self):
        _, _, _, dr = _parse_argv(["prog", "--hp", "cfg.yaml", "--dry-run"])
        assert dr

    def test_exits_without_config(self):
        with pytest.raises(SystemExit) as exc:
            _parse_argv(["prog", "a=1"])
        assert exc.value.code == 1

    def test_rejects_unknown_flag(self):
        with pytest.raises(SystemExit):
            _parse_argv(["prog", "--hp", "cfg.yaml", "--whatever"])

    def test_hp_without_value_errors(self):
        with pytest.raises(SystemExit):
            _parse_argv(["prog", "--hp"])


# --------------------------------------------------------------------------- #
# _expand_sweeps
# --------------------------------------------------------------------------- #
class TestExpandSweeps:
    def test_no_multirun_passes_through(self):
        assert _expand_sweeps(["a=1", "b=2"], multirun=False) == [["a=1", "b=2"]]

    def test_no_multirun_rejects_sweeps(self):
        with pytest.raises(ValueError, match="requires --multirun"):
            _expand_sweeps(["a=1,2"], multirun=False)

    def test_multirun_expands_cross_product(self):
        combos = _expand_sweeps(["a=1,2", "b=x,y"], multirun=True)
        assert len(combos) == 4
        # Every combination contains exactly one a-value and one b-value.
        as_sets = {frozenset(c) for c in combos}
        assert as_sets == {
            frozenset({"a=1", "b=x"}),
            frozenset({"a=1", "b=y"}),
            frozenset({"a=2", "b=x"}),
            frozenset({"a=2", "b=y"}),
        }

    def test_multirun_with_no_sweep_yields_single_combo(self):
        assert _expand_sweeps(["a=1", "b=2"], multirun=True) == [["a=1", "b=2"]]


# --------------------------------------------------------------------------- #
# validate_config
# --------------------------------------------------------------------------- #
class TestValidateConfig:
    def _make_config(self, extra: dict | None = None):
        base = {"config_schema_kind": "noether.core.schemas.schema.ConfigSchema"}
        if extra:
            base.update(extra)
        return OmegaConf.create(base)

    def test_calls_class_constructor_with_schema_kind(self):
        config = self._make_config()
        mock_schema_instance = MagicMock()
        mock_schema_class = MagicMock(return_value=mock_schema_instance)

        with patch(_MODULE_PATH + ".class_constructor_from_class_path", return_value=mock_schema_class) as mock_ctor:
            result = validate_config(config)

        mock_ctor.assert_called_once_with("noether.core.schemas.schema.ConfigSchema")
        mock_schema_class.assert_called_once()
        assert result is mock_schema_instance

    def test_propagates_validation_error_from_schema(self):
        config = self._make_config()
        mock_schema_class = MagicMock(side_effect=ValueError("bad field"))
        with patch(_MODULE_PATH + ".class_constructor_from_class_path", return_value=mock_schema_class):
            with pytest.raises(ValueError, match="bad field"):
                validate_config(config)


# --------------------------------------------------------------------------- #
# _validate_all_combos
# --------------------------------------------------------------------------- #
class TestValidateAllCombos:
    def _validated(self, slurm: SlurmConfig | None) -> MagicMock:
        m = MagicMock()
        m.slurm = slurm
        return m

    def test_returns_first_slurm_when_all_match(self):
        slurm = SlurmConfig(name="job", slurm_partition="gpu")
        combos = [["a=1"], ["a=2"]]

        with (
            patch(_MODULE_PATH + "._compose_config"),
            patch(_MODULE_PATH + ".validate_config", return_value=self._validated(slurm)),
        ):
            result = _validate_all_combos(Path("/cfg.yaml"), combos)

        assert result == slurm

    def test_aborts_when_validation_fails_on_combo(self, capsys):
        with (
            patch(_MODULE_PATH + "._compose_config"),
            patch(_MODULE_PATH + ".validate_config", side_effect=[self._validated(SlurmConfig()), ValueError("boom")]),
            pytest.raises(SystemExit) as exc,
        ):
            _validate_all_combos(Path("/cfg.yaml"), [["a=1"], ["a=2"]])

        assert exc.value.code == 1
        assert "boom" in capsys.readouterr().err

    def test_rejects_missing_slurm_section(self, capsys):
        with (
            patch(_MODULE_PATH + "._compose_config"),
            patch(_MODULE_PATH + ".validate_config", return_value=self._validated(None)),
            pytest.raises(SystemExit) as exc,
        ):
            _validate_all_combos(Path("/cfg.yaml"), [[]])

        assert exc.value.code == 1
        assert "SLURM configuration is required" in capsys.readouterr().err

    def test_rejects_sweep_over_slurm_fields(self, capsys):
        slurm_a = SlurmConfig(name="job", slurm_partition="gpu")
        slurm_b = SlurmConfig(name="job", slurm_partition="cpu")  # different!
        with (
            patch(_MODULE_PATH + "._compose_config"),
            patch(
                _MODULE_PATH + ".validate_config",
                side_effect=[self._validated(slurm_a), self._validated(slurm_b)],
            ),
            pytest.raises(SystemExit) as exc,
        ):
            _validate_all_combos(Path("/cfg.yaml"), [["a=1"], ["a=2"]])

        assert exc.value.code == 1
        assert "sweeping over fields under 'slurm'" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# _format_task_preview
# --------------------------------------------------------------------------- #
class TestFormatTaskPreview:
    def test_no_overrides(self):
        assert _format_task_preview(Path("/abs/cfg.yaml"), []) == "noether-train --hp /abs/cfg.yaml"

    def test_with_overrides(self):
        assert _format_task_preview(Path("/abs/cfg.yaml"), ["a=1", "b=2"]) == "noether-train --hp /abs/cfg.yaml a=1 b=2"

    def test_handles_path_with_spaces(self):
        assert _format_task_preview(Path("/my projects/cfg.yaml"), []) == "noether-train --hp /my projects/cfg.yaml"


# --------------------------------------------------------------------------- #
# _train_entrypoint
# --------------------------------------------------------------------------- #
class TestTrainEntrypoint:
    def test_is_picklable(self):
        # submitit serialises the submitted function to disk for the worker to load.
        assert pickle.loads(pickle.dumps(_train_entrypoint)) is _train_entrypoint

    def test_sets_up_argv_and_cwd_then_calls_main_train(self, tmp_path: Path, monkeypatch):
        # main_train calls setup_hydra() at import time, which inspects sys.argv.
        # _train_entrypoint must therefore set up sys.argv *before* importing main_train.
        # We assert this by capturing sys.argv at the moment our fake main() is called.
        captured: dict = {}

        def _record_state():
            captured["argv"] = list(sys.argv)
            captured["cwd"] = os.getcwd()

        fake_main_train = MagicMock()
        fake_main_train.main = MagicMock(side_effect=_record_state)
        monkeypatch.setitem(sys.modules, "noether.training.cli.main_train", fake_main_train)

        original_cwd = os.getcwd()
        original_argv = list(sys.argv)
        try:
            cfg_path = tmp_path / "cfg.yaml"
            _train_entrypoint(str(cfg_path), ["a=1", "b=2"], str(tmp_path))
        finally:
            os.chdir(original_cwd)
            sys.argv[:] = original_argv

        fake_main_train.main.assert_called_once_with()
        assert captured["argv"][1:] == ["--hp", str(cfg_path), "a=1", "b=2"]
        assert captured["cwd"] == str(tmp_path)
        assert str(tmp_path) in sys.path


# --------------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------------- #
class TestMain:
    @pytest.fixture
    def slurm(self) -> SlurmConfig:
        return SlurmConfig(name="job", slurm_partition="gpu", timeout_min=60)

    @pytest.fixture
    def cfg_path(self, tmp_path: Path) -> Path:
        p = tmp_path / "cfg.yaml"
        p.touch()
        return p

    def _run(self, argv: list[str], slurm: SlurmConfig, *, mock_jobs: list | None = None):
        """Drive main() with a stubbed-out validation pipeline and submitit."""
        executor = MagicMock()
        executor.batch.return_value.__enter__ = MagicMock(return_value=None)
        executor.batch.return_value.__exit__ = MagicMock(return_value=False)
        if mock_jobs is None:
            executor.submit.side_effect = lambda *args, **kwargs: MagicMock(job_id="42_0")
        else:
            executor.submit.side_effect = mock_jobs

        with (
            patch.object(__import__("sys"), "argv", argv),
            patch(_MODULE_PATH + "._validate_all_combos", return_value=slurm),
            patch(_MODULE_PATH + ".submitit.AutoExecutor", return_value=executor) as ex_cls,
        ):
            try:
                main()
            except SystemExit as e:
                return ex_cls, executor, e.code
        return ex_cls, executor, None

    def test_help_exits_cleanly(self, capsys):
        with patch.object(__import__("sys"), "argv", ["prog", "--help"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 0
        assert "noether-train-submit-job" in capsys.readouterr().out

    def test_missing_config_file_errors(self, slurm):
        argv = ["prog", "--hp", "/does/not/exist.yaml"]
        with (
            patch.object(__import__("sys"), "argv", argv),
            pytest.raises(SystemExit) as exc,
        ):
            main()
        assert exc.value.code == 1

    def test_single_run_submits_once(self, slurm, cfg_path):
        argv = ["prog", "--hp", str(cfg_path)]
        ex_cls, executor, _ = self._run(argv, slurm)

        ex_cls.assert_called_once_with(folder=slurm.folder)
        executor.update_parameters.assert_called_once()
        params = executor.update_parameters.call_args.kwargs
        assert params["name"] == "job"
        assert params["slurm_partition"] == "gpu"
        assert params["timeout_min"] == 60
        executor.submit.assert_called_once()
        executor.batch.assert_not_called()  # batch() only used for arrays
        # submit(_train_entrypoint, str(config_path), overrides, cwd)
        fn, submitted_path, overrides, cwd = executor.submit.call_args.args
        assert fn is _train_entrypoint
        assert submitted_path == str(cfg_path)
        assert overrides == []
        assert cwd == os.getcwd()

    def test_multirun_uses_batch_and_submits_n_times(self, slurm, cfg_path):
        argv = ["prog", "--hp", str(cfg_path), "-m", "+seed=1,2,3"]
        jobs = [MagicMock(job_id=f"42_{i}") for i in range(3)]
        ex_cls, executor, _ = self._run(argv, slurm, mock_jobs=jobs)

        executor.batch.assert_called_once()
        assert executor.submit.call_count == 3
        seeds_seen: set[str] = set()
        for call in executor.submit.call_args_list:
            fn, submitted_path, overrides, _cwd = call.args
            assert fn is _train_entrypoint
            assert submitted_path == str(cfg_path)
            seeds_seen.update(o for o in overrides if o.startswith("+seed="))
        assert seeds_seen == {"+seed=1", "+seed=2", "+seed=3"}

    def test_dry_run_does_not_construct_executor(self, slurm, cfg_path, capsys):
        argv = ["prog", "--hp", str(cfg_path), "--dry-run"]
        with (
            patch.object(__import__("sys"), "argv", argv),
            patch(_MODULE_PATH + "._validate_all_combos", return_value=slurm),
            patch(_MODULE_PATH + ".submitit.AutoExecutor") as ex_cls,
            pytest.raises(SystemExit) as exc,
        ):
            main()

        assert exc.value.code == 0
        ex_cls.assert_not_called()
        out = capsys.readouterr().out
        assert "[dry-run]" in out
        assert "noether-train" in out

    def test_dry_run_multirun_lists_all_tasks(self, slurm, cfg_path, capsys):
        argv = ["prog", "--hp", str(cfg_path), "--dry-run", "-m", "+seed=1,2"]
        with (
            patch.object(__import__("sys"), "argv", argv),
            patch(_MODULE_PATH + "._validate_all_combos", return_value=slurm),
            patch(_MODULE_PATH + ".submitit.AutoExecutor"),
            pytest.raises(SystemExit),
        ):
            main()

        out = capsys.readouterr().out
        assert "Would submit 2 task(s)" in out
        assert "+seed=1" in out
        assert "+seed=2" in out


# --------------------------------------------------------------------------- #
# SlurmConfig.to_executor_kwargs
# --------------------------------------------------------------------------- #
class TestSlurmConfigToExecutorKwargs:
    def test_excludes_none_fields(self):
        folder, params = SlurmConfig(name="job", slurm_partition="gpu").to_executor_kwargs()
        assert folder == "submitit_logs"
        assert params == {"name": "job", "slurm_partition": "gpu", "timeout_min": 0}

    def test_folder_is_returned_separately(self):
        folder, params = SlurmConfig(folder="/tmp/logs").to_executor_kwargs()
        assert folder == "/tmp/logs"
        assert "folder" not in params

    def test_passes_setup_and_additional_parameters(self):
        cfg = SlurmConfig(
            slurm_setup=["source .venv/bin/activate"],
            slurm_additional_parameters={"nice": 0, "reservation": "res"},
        )
        _, params = cfg.to_executor_kwargs()
        assert params["slurm_setup"] == ["source .venv/bin/activate"]
        assert params["slurm_additional_parameters"] == {"nice": 0, "reservation": "res"}
