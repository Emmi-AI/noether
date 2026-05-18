#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# We need to patch setup_hydra BEFORE importing the module to avoid running it:
with (
    patch("noether.training.cli.setup_hydra"),
    patch("noether.inference.cli.main_inference._inject_hp_resolved_into_argv"),
):
    from noether.inference.cli import main_inference
    from noether.inference.cli.main_inference import (
        _build_run_dir,
        _pop_eval_path_args,
    )
    from noether.inference.cli.main_inference import main as inference_main

_MODULE_PATH = "noether.inference.cli.main_inference"


class TestPopEvalPathArgs:
    """`_pop_eval_path_args` extracts navigation args without polluting Hydra overrides."""

    def test_pops_run_dir(self):
        popped, remaining = _pop_eval_path_args(["run_dir=outputs/abc/train", "tracker=disabled"])
        assert popped == {"run_dir": "outputs/abc/train"}
        assert remaining == ["tracker=disabled"]

    def test_pops_run_dir_with_plus_prefix(self):
        popped, remaining = _pop_eval_path_args(["+run_dir=outputs/abc/train"])
        assert popped == {"run_dir": "outputs/abc/train"}
        assert remaining == []

    def test_pops_legacy_trio_when_input_dir_present(self):
        popped, remaining = _pop_eval_path_args(
            ["input_dir=outputs", "run_id=abc", "stage_name=train", "tracker=disabled"]
        )
        assert popped == {"input_dir": "outputs", "run_id": "abc", "stage_name": "train"}
        assert remaining == ["tracker=disabled"]

    def test_does_not_pop_run_id_alone(self):
        """Without input_dir, `run_id=foo` is a normal config override (changes the
        eval output dir), not a path-navigation arg."""
        popped, remaining = _pop_eval_path_args(["run_dir=outputs/abc/train", "run_id=eval_run"])
        assert popped == {"run_dir": "outputs/abc/train"}
        assert remaining == ["run_id=eval_run"]


class TestBuildRunDir:
    def test_run_dir_takes_priority(self):
        result = _build_run_dir({"run_dir": "/abs/path", "input_dir": "should_be_ignored"})
        assert result == Path("/abs/path").resolve()

    def test_legacy_form_assembles_path(self):
        result = _build_run_dir({"input_dir": "/root", "run_id": "abc", "stage_name": "train"})
        assert result == Path("/root/abc/train").resolve()

    def test_legacy_form_without_stage_name(self):
        result = _build_run_dir({"input_dir": "/root", "run_id": "abc"})
        assert result == Path("/root/abc").resolve()

    def test_returns_none_when_no_path_args(self):
        assert _build_run_dir({}) is None
        assert _build_run_dir({"input_dir": "/root"}) is None  # missing run_id


class TestInjectHpResolved:
    """`_inject_hp_resolved_into_argv` rewrites argv to point Hydra at the run's hp_resolved.yaml."""

    def _make_run_dir(self, tmp_path, train_config):
        """Create a fake run_dir with a real hp_resolved.yaml on disk."""
        run_dir = tmp_path / "outputs" / "2026-04-16_g5s7p"
        run_dir.mkdir(parents=True)
        (run_dir / "hp_resolved.yaml").write_text(yaml.safe_dump(train_config))
        return run_dir

    def test_injects_hp_arg_for_run_dir(self, tmp_path, monkeypatch):
        # Include a tuple so we can confirm sanitization strips `!!python/tuple`.
        run_dir = self._make_run_dir(tmp_path, {"run_id": "abc", "shape": (1, 2, 3), "output_path": "/source/out"})
        monkeypatch.setattr(sys, "argv", ["noether-eval", f"run_dir={run_dir}", "tracker=disabled"])

        main_inference._inject_hp_resolved_into_argv()

        assert sys.argv[0] == "noether-eval"
        assert sys.argv[1] == "--hp"
        # The injected path is a sanitized copy in a temp dir, not the original.
        injected = Path(sys.argv[2])
        assert injected.name == "hp_resolved.yaml"
        assert injected != (run_dir / "hp_resolved.yaml")
        sanitized = injected.read_text()
        assert "!!python/tuple" not in sanitized
        assert "shape:" in sanitized  # tuple round-tripped as a list
        # Forced overrides for run_id / stage_name / resume_* come before user args. run_id is single-quoted so
        # Hydra/OmegaConf keeps it as a string (slurm job ids are all digits and would otherwise parse as int).
        assert "++run_id='abc'" in sys.argv
        assert "++resume_run_id='abc'" in sys.argv
        # The source's output_path is pinned as resume_output_path so eval can
        # safely override `output_path=...` without breaking checkpoint lookup.
        assert "++resume_output_path=/source/out" in sys.argv
        assert "tracker=disabled" in sys.argv
        # User-supplied args appear after the injected overrides.
        assert sys.argv.index("tracker=disabled") > sys.argv.index("++resume_run_id='abc'")

    def test_skips_resume_output_path_when_hp_lacks_it(self, tmp_path, monkeypatch):
        """If hp_resolved.yaml has no ``output_path`` (e.g. dumped before this
        field was emitted), don't inject ``++resume_output_path=`` — the runner
        falls back to ``output_path`` for the resume lookup."""
        run_dir = self._make_run_dir(tmp_path, {"run_id": "abc"})
        monkeypatch.setattr(sys, "argv", ["noether-eval", f"run_dir={run_dir}"])

        main_inference._inject_hp_resolved_into_argv()

        assert not any(a.startswith("++resume_output_path=") for a in sys.argv)

    def test_infers_run_id_from_path_when_hp_lacks_it(self, tmp_path, monkeypatch):
        """When hp_resolved.yaml omits run_id (training-time generated), the run id
        is inferred from the run_dir name."""
        run_dir = self._make_run_dir(tmp_path, {})
        monkeypatch.setattr(sys, "argv", ["noether-eval", f"run_dir={run_dir}"])

        main_inference._inject_hp_resolved_into_argv()

        # run_dir.name == "2026-04-16_g5s7p" per `_make_run_dir`
        assert "++run_id='2026-04-16_g5s7p'" in sys.argv
        assert "++resume_run_id='2026-04-16_g5s7p'" in sys.argv
        assert "++stage_name=" in sys.argv  # empty stage_name when not in config

    def test_infers_run_id_from_parent_when_stage_name_present(self, tmp_path, monkeypatch):
        """When hp_resolved.yaml has stage_name, run_dir = output_path/run_id/stage_name,
        so run_id comes from run_dir.parent.name."""
        run_dir = tmp_path / "outputs" / "my_run" / "train"
        run_dir.mkdir(parents=True)
        (run_dir / "hp_resolved.yaml").write_text(yaml.safe_dump({"stage_name": "train"}))
        monkeypatch.setattr(sys, "argv", ["noether-eval", f"run_dir={run_dir}"])

        main_inference._inject_hp_resolved_into_argv()

        assert "++run_id='my_run'" in sys.argv
        assert "++stage_name=train" in sys.argv

    def test_no_op_when_user_supplies_hp(self, tmp_path, monkeypatch):
        original = ["noether-eval", "--hp", "configs/eval.yaml", "run_dir=outputs/abc"]
        monkeypatch.setattr(sys, "argv", list(original))

        main_inference._inject_hp_resolved_into_argv()

        assert sys.argv == original

    def test_no_op_for_help_flag(self, monkeypatch):
        original = ["noether-eval", "--help"]
        monkeypatch.setattr(sys, "argv", list(original))

        main_inference._inject_hp_resolved_into_argv()

        assert sys.argv == original

    def test_raises_when_hp_resolved_missing(self, tmp_path, monkeypatch):
        bad_dir = tmp_path / "missing_run"
        bad_dir.mkdir()
        monkeypatch.setattr(sys, "argv", ["noether-eval", f"run_dir={bad_dir}"])

        with pytest.raises(FileNotFoundError, match="hp_resolved.yaml not found"):
            main_inference._inject_hp_resolved_into_argv()

    def test_no_op_when_no_path_args(self, monkeypatch):
        """If neither run_dir nor the legacy trio is present, leave argv alone
        and let setup_hydra / hydra produce a clear error downstream."""
        original = ["noether-eval", "tracker=disabled"]
        monkeypatch.setattr(sys, "argv", list(original))

        main_inference._inject_hp_resolved_into_argv()

        assert sys.argv == original


@patch(_MODULE_PATH + ".InferenceRunner")
@patch(_MODULE_PATH + ".OmegaConf")
@patch(_MODULE_PATH + ".sys")
@patch(_MODULE_PATH + ".os")
@patch(_MODULE_PATH + ".hydra")
class TestMain:
    """`main()` runs after Hydra has already loaded hp_resolved.yaml as the base config."""

    def test_dispatches_to_inference_runner(self, mock_hydra, mock_os, mock_sys, mock_omegaconf, mock_runner_cls):
        mock_hydra.utils.get_original_cwd.return_value = "/cwd"
        mock_omegaconf.to_container.return_value = {"final": "config"}

        # `resume_run_id` and `resume_checkpoint` were injected into argv during
        # `_inject_hp_resolved_into_argv`, so by the time main() runs Hydra has
        # them on the config.
        eval_config = MagicMock()
        eval_config.get.side_effect = lambda key, default=None: {
            "resume_run_id": "train_run_id_123",
            "resume_checkpoint": "latest",
        }.get(key, default)

        inference_main.__wrapped__(eval_config)

        mock_runner_cls.return_value.run.assert_called_once_with({"final": "config"})

    def test_missing_resume_run_id_raises(self, mock_hydra, mock_os, mock_sys, mock_omegaconf, mock_runner_cls):
        mock_hydra.utils.get_original_cwd.return_value = "/cwd"

        eval_config = MagicMock()
        eval_config.get.return_value = None

        with pytest.raises(ValueError, match="run_dir"):
            inference_main.__wrapped__(eval_config)
