#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import sys
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from noether.core.schemas.slurm import SlurmConfig
from noether.training.cli.submit_job import _collect_hydra_overrides, _find_config_path, validate_config

_MODULE_PATH = "noether.training.cli.submit_job"


class TestValidateConfig:
    def _make_config(self, extra: dict | None = None):
        base = {"config_schema_kind": "noether.core.schemas.schema.ConfigSchema"}
        if extra:
            base.update(extra)
        return OmegaConf.create(base)

    def test_raises_if_config_schema_kind_missing(self):
        config = OmegaConf.create({"some_key": "some_value"})
        with pytest.raises(ValueError, match="config_schema_kind"):
            validate_config(config)

    def test_raises_if_config_schema_kind_empty(self):
        config = OmegaConf.create({"config_schema_kind": ""})
        with pytest.raises(ValueError, match="config_schema_kind"):
            validate_config(config)

    def test_calls_class_constructor_with_schema_kind(self):
        config = self._make_config()
        mock_schema_instance = MagicMock()
        mock_schema_class = MagicMock(return_value=mock_schema_instance)

        with patch(_MODULE_PATH + ".class_constructor_from_class_path", return_value=mock_schema_class) as mock_ctor:
            result = validate_config(config)

        mock_ctor.assert_called_once_with("noether.core.schemas.schema.ConfigSchema")
        mock_schema_class.assert_called_once()
        assert result is mock_schema_instance

    def test_raises_if_class_constructor_fails(self):
        config = self._make_config()
        with patch(_MODULE_PATH + ".class_constructor_from_class_path", side_effect=ImportError("module not found")):
            with pytest.raises(ImportError, match="module not found"):
                validate_config(config)

    def test_propagates_validation_error_from_schema(self):
        config = self._make_config()
        mock_schema_class = MagicMock(side_effect=ValueError("bad field"))

        with patch(_MODULE_PATH + ".class_constructor_from_class_path", return_value=mock_schema_class):
            with pytest.raises(ValueError, match="bad field"):
                validate_config(config)


class TestFindConfigPath:
    def _run(self, argv, raw_config_path):
        with (
            patch.object(sys, "argv", argv),
            patch(_MODULE_PATH + "._RAW_CONFIG_PATH", raw_config_path),
        ):
            return _find_config_path()

    def test_returns_absolute_path_from_raw_config_path(self, tmp_path):
        config_file = tmp_path / "train.yaml"
        config_file.touch()
        result = self._run(["prog"], raw_config_path=str(config_file))
        assert result == str(config_file.resolve())

    def test_resolves_relative_raw_config_path_to_absolute(self, tmp_path, monkeypatch):
        config_file = tmp_path / "train.yaml"
        config_file.touch()
        monkeypatch.chdir(tmp_path)
        result = self._run(["prog"], raw_config_path="train.yaml")
        assert result == str(config_file.resolve())

    def test_finds_path_from_cp_and_cn_when_raw_is_none(self, tmp_path):
        config_file = tmp_path / "train.yaml"
        config_file.touch()
        result = self._run(
            ["prog", "-cp", str(tmp_path), "-cn", "train"],
            raw_config_path=None,
        )
        assert result == str(config_file.resolve())

    def test_cp_cn_appends_yaml_extension_if_missing(self, tmp_path):
        config_file = tmp_path / "experiment.yaml"
        config_file.touch()
        result = self._run(
            ["prog", "-cp", str(tmp_path), "-cn", "experiment"],
            raw_config_path=None,
        )
        assert result.endswith(".yaml")

    def test_cp_cn_does_not_double_append_yaml(self, tmp_path):
        config_file = tmp_path / "experiment.yaml"
        config_file.touch()
        result = self._run(
            ["prog", "-cp", str(tmp_path), "-cn", "experiment.yaml"],
            raw_config_path=None,
        )
        assert not result.endswith(".yaml.yaml")

    def test_exits_if_no_config_found(self):
        with pytest.raises(SystemExit) as exc_info:
            self._run(["prog", "some_random_arg=value"], raw_config_path=None)
        assert exc_info.value.code == 1

    def test_raw_config_path_takes_priority_over_cp_cn(self, tmp_path):
        direct = tmp_path / "direct.yaml"
        direct.touch()
        other = tmp_path / "other.yaml"
        other.touch()
        result = self._run(
            ["prog", "-cp", str(tmp_path), "-cn", "other"],
            raw_config_path=str(direct),
        )
        assert result == str(direct.resolve())


class TestCollectHydraOverrides:
    def _run_with_argv(self, argv):
        with patch.object(sys, "argv", argv):
            return _collect_hydra_overrides()

    def test_collects_key_value_overrides(self):
        result = self._run_with_argv(["prog", "--hp", "cfg.yaml", "seed=42", "trainer.lr=0.01"])
        assert result == ["seed=42", "trainer.lr=0.01"]

    def test_skips_hp_flag_and_its_value(self):
        result = self._run_with_argv(["prog", "--hp", "cfg.yaml", "key=val"])
        assert "--hp" not in result
        assert "cfg.yaml" not in result

    def test_skips_cp_flag_and_its_value(self):
        result = self._run_with_argv(["prog", "-cp", "some/dir", "key=val"])
        assert "-cp" not in result
        assert "some/dir" not in result

    def test_skips_cn_flag_and_its_value(self):
        result = self._run_with_argv(["prog", "-cn", "train", "key=val"])
        assert "-cn" not in result
        assert "train" not in result

    def test_skips_hydra_prefixed_overrides(self):
        result = self._run_with_argv(["prog", "--hp", "cfg.yaml", "hydra.run.dir=.", "key=val"])
        assert "hydra.run.dir=." not in result
        assert "key=val" in result

    def test_skips_dash_prefixed_args(self):
        result = self._run_with_argv(["prog", "--hp", "cfg.yaml", "--verbose", "key=val"])
        assert "--verbose" not in result

    def test_returns_empty_list_when_no_overrides(self):
        result = self._run_with_argv(["prog", "--hp", "cfg.yaml"])
        assert result == []

    def test_collects_multiple_overrides(self):
        result = self._run_with_argv(["prog", "--hp", "cfg.yaml", "a=1", "b=2", "c=3"])
        assert result == ["a=1", "b=2", "c=3"]


class TestMain:
    @pytest.fixture
    def mock_validated_config(self):
        config = MagicMock()
        config.slurm = SlurmConfig(job_name="test-job", partition="gpu", time="01:00:00")
        return config

    def _call_main(self, mock_config, dry_run=False):
        from noether.training.cli.submit_job import main

        with patch(_MODULE_PATH + "._DRY_RUN", dry_run):
            main.__wrapped__(mock_config)

    def test_exits_on_validation_failure(self):
        mock_config = OmegaConf.create({})
        with patch(_MODULE_PATH + ".validate_config", side_effect=ValueError("bad config")):
            with pytest.raises(SystemExit) as exc_info:
                self._call_main(mock_config)
        assert exc_info.value.code == 1

    def test_raises_if_slurm_config_missing(self, mock_validated_config):
        mock_validated_config.slurm = None
        mock_config = OmegaConf.create({})
        with patch(_MODULE_PATH + ".validate_config", return_value=mock_validated_config):
            with pytest.raises(SystemExit) as exc_info:
                self._call_main(mock_config)
        assert exc_info.value.code == 1

    def test_submits_sbatch_command(self, mock_validated_config):
        mock_config = OmegaConf.create({})
        with (
            patch(_MODULE_PATH + ".validate_config", return_value=mock_validated_config),
            patch(_MODULE_PATH + "._find_config_path", return_value="/abs/configs/train.yaml"),
            patch(_MODULE_PATH + "._collect_hydra_overrides", return_value=[]),
            patch(_MODULE_PATH + ".subprocess.run", return_value=MagicMock(returncode=0)) as mock_run,
            pytest.raises(SystemExit) as exc_info,
        ):
            self._call_main(mock_config)

        assert exc_info.value.code == 0
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "sbatch" in cmd
        assert "--wrap=" in cmd
        assert "noether-train" in cmd
        assert "/abs/configs/train.yaml" in cmd

    def test_config_path_is_shell_quoted(self, mock_validated_config):
        mock_config = OmegaConf.create({})
        spaced_path = "/my projects/train config.yaml"
        with (
            patch(_MODULE_PATH + ".validate_config", return_value=mock_validated_config),
            patch(_MODULE_PATH + "._find_config_path", return_value=spaced_path),
            patch(_MODULE_PATH + "._collect_hydra_overrides", return_value=[]),
            patch(_MODULE_PATH + ".subprocess.run", return_value=MagicMock(returncode=0)) as mock_run,
            pytest.raises(SystemExit),
        ):
            self._call_main(mock_config)

        cmd = mock_run.call_args[0][0]
        assert spaced_path in cmd

    def test_includes_hydra_overrides_in_train_cmd(self, mock_validated_config):
        mock_config = OmegaConf.create({})
        with (
            patch(_MODULE_PATH + ".validate_config", return_value=mock_validated_config),
            patch(_MODULE_PATH + "._find_config_path", return_value="/cfg.yaml"),
            patch(_MODULE_PATH + "._collect_hydra_overrides", return_value=["seed=42", "trainer.lr=0.001"]),
            patch(_MODULE_PATH + ".subprocess.run", return_value=MagicMock(returncode=0)) as mock_run,
            pytest.raises(SystemExit),
        ):
            self._call_main(mock_config)

        cmd = mock_run.call_args[0][0]
        assert "seed=42" in cmd
        assert "trainer.lr=0.001" in cmd

    def test_overrides_with_single_quotes_are_shell_safe(self, mock_validated_config):
        mock_config = OmegaConf.create({})
        override = "name='my run'"
        with (
            patch(_MODULE_PATH + ".validate_config", return_value=mock_validated_config),
            patch(_MODULE_PATH + "._find_config_path", return_value="/cfg.yaml"),
            patch(_MODULE_PATH + "._collect_hydra_overrides", return_value=[override]),
            patch(_MODULE_PATH + ".subprocess.run", return_value=MagicMock(returncode=0)) as mock_run,
            pytest.raises(SystemExit),
        ):
            self._call_main(mock_config)

        cmd = mock_run.call_args[0][0]
        assert override in cmd

    def test_chdir_appears_in_sbatch_args_not_as_cd_prefix(self, mock_validated_config):
        mock_validated_config.slurm = SlurmConfig(job_name="job", chdir="/workspace/project")
        mock_config = OmegaConf.create({})
        with (
            patch(_MODULE_PATH + ".validate_config", return_value=mock_validated_config),
            patch(_MODULE_PATH + "._find_config_path", return_value="/cfg.yaml"),
            patch(_MODULE_PATH + "._collect_hydra_overrides", return_value=[]),
            patch(_MODULE_PATH + ".subprocess.run", return_value=MagicMock(returncode=0)) as mock_run,
            pytest.raises(SystemExit),
        ):
            self._call_main(mock_config)

        cmd = mock_run.call_args[0][0]
        assert "--chdir=/workspace/project" in cmd
        assert "cd /workspace/project;" not in cmd

    def test_prepends_source_when_env_path_set(self, mock_validated_config):
        mock_validated_config.slurm = SlurmConfig(job_name="job", env_path="/opt/venv/bin/activate")
        mock_config = OmegaConf.create({})
        with (
            patch(_MODULE_PATH + ".validate_config", return_value=mock_validated_config),
            patch(_MODULE_PATH + "._find_config_path", return_value="/cfg.yaml"),
            patch(_MODULE_PATH + "._collect_hydra_overrides", return_value=[]),
            patch(_MODULE_PATH + ".subprocess.run", return_value=MagicMock(returncode=0)) as mock_run,
            pytest.raises(SystemExit),
        ):
            self._call_main(mock_config)

        cmd = mock_run.call_args[0][0]
        assert "source" in cmd
        assert "/opt/venv/bin/activate" in cmd

    def test_no_source_when_env_path_not_set(self, mock_validated_config):
        mock_validated_config.slurm = SlurmConfig(job_name="job")
        mock_config = OmegaConf.create({})
        with (
            patch(_MODULE_PATH + ".validate_config", return_value=mock_validated_config),
            patch(_MODULE_PATH + "._find_config_path", return_value="/cfg.yaml"),
            patch(_MODULE_PATH + "._collect_hydra_overrides", return_value=[]),
            patch(_MODULE_PATH + ".subprocess.run", return_value=MagicMock(returncode=0)) as mock_run,
            pytest.raises(SystemExit),
        ):
            self._call_main(mock_config)

        cmd = mock_run.call_args[0][0]
        assert "source " not in cmd

    def test_propagates_nonzero_returncode(self, mock_validated_config):
        mock_config = OmegaConf.create({})
        with (
            patch(_MODULE_PATH + ".validate_config", return_value=mock_validated_config),
            patch(_MODULE_PATH + "._find_config_path", return_value="/cfg.yaml"),
            patch(_MODULE_PATH + "._collect_hydra_overrides", return_value=[]),
            patch(_MODULE_PATH + ".subprocess.run", return_value=MagicMock(returncode=42)),
            pytest.raises(SystemExit) as exc_info,
        ):
            self._call_main(mock_config)

        assert exc_info.value.code == 42

    def test_dry_run_prints_command_without_submitting(self, mock_validated_config, capsys):
        mock_config = OmegaConf.create({})
        with (
            patch(_MODULE_PATH + ".validate_config", return_value=mock_validated_config),
            patch(_MODULE_PATH + "._find_config_path", return_value="/cfg.yaml"),
            patch(_MODULE_PATH + "._collect_hydra_overrides", return_value=[]),
            patch(_MODULE_PATH + ".subprocess.run") as mock_run,
            pytest.raises(SystemExit) as exc_info,
        ):
            self._call_main(mock_config, dry_run=True)

        assert exc_info.value.code == 0
        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "[dry-run]" in captured.out
        assert "sbatch" in captured.out
