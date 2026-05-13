#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import os
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import RootModel

from noether.core.configs.hyperparameters import Hyperparameters
from noether.core.schemas.callbacks import OnlineLossCallbackConfig
from noether.core.schemas.schema import ConfigSchema
from noether.inference.evaluate import evaluate

_INFERENCE_RUNNER_MAIN = "noether.inference.evaluate.InferenceRunner.main"


class _DimSpec(RootModel[OrderedDict[str, int]]):
    pass


class _EvalHP(ConfigSchema):
    """Minimal ConfigSchema subclass with a non-trivial polymorphic field
    suitable for testing override-path navigation."""

    spec: _DimSpec


@pytest.fixture
def hp_resolved_dir(tmp_path: Path) -> Path:
    """Create a directory with a saved ``hp_resolved.yaml`` representing a
    training run that ``evaluate`` can target."""
    os.environ["MASTER_PORT"] = "12345"
    params = _EvalHP(
        output_path=str(tmp_path),
        run_id="run_abc123",
        datasets=dict(),
        model=dict(name="m", kind="k"),
        trainer=dict(kind="mock", effective_batch_size=4, callbacks=[], max_epochs=1),
        spec=_DimSpec({"a": 1, "b": 2}),
    )
    run_dir = tmp_path / "run_abc123"
    run_dir.mkdir()
    Hyperparameters.save_resolved(params, run_dir / "hp_resolved.yaml")
    return run_dir


class TestEvaluate:
    """``evaluate`` should mirror the ``noether-eval`` CLI: load the saved
    config, wire ``resume_*`` against the run dir, optionally swap the callback
    list, and dispatch via ``InferenceRunner.main``."""

    def test_dispatches_to_inference_runner(self, hp_resolved_dir: Path):
        with patch(_INFERENCE_RUNNER_MAIN) as mock_main:
            evaluate(hp_resolved_dir, resume_checkpoint="best.metric")

        mock_main.assert_called_once()
        kwargs = mock_main.call_args.kwargs
        assert kwargs["device"] == "cuda"
        config = kwargs["config"]
        # resume_* points back at the saved training run
        assert config.resume_run_id == "run_abc123"
        assert config.resume_checkpoint == "best.metric"
        assert config.resume_output_path == Path(config.output_path)

    def test_replaces_callbacks(self, hp_resolved_dir: Path):
        eval_cb = OnlineLossCallbackConfig(every_n_epochs=1)
        with patch(_INFERENCE_RUNNER_MAIN) as mock_main:
            evaluate(hp_resolved_dir, callbacks=[eval_cb])

        config = mock_main.call_args.kwargs["config"]
        assert config.trainer.callbacks == [eval_cb]

    def test_overrides_stage_name_for_eval_outputs(self, hp_resolved_dir: Path):
        """``stage_name`` parameter routes outputs to a sub-stage while
        ``resume_stage_name`` keeps pointing at the training-time stage."""
        with patch(_INFERENCE_RUNNER_MAIN) as mock_main:
            evaluate(hp_resolved_dir, stage_name="eval_sub")

        config = mock_main.call_args.kwargs["config"]
        assert config.stage_name == "eval_sub"
        # The source run's stage_name was empty — resume should still target it.
        assert config.resume_stage_name == ""

    def test_disable_tracker(self, hp_resolved_dir: Path):
        with patch(_INFERENCE_RUNNER_MAIN) as mock_main:
            evaluate(hp_resolved_dir, disable_tracker=True)

        config = mock_main.call_args.kwargs["config"]
        assert config.tracker is None

    def test_missing_run_dir(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="hp_resolved.yaml"):
            evaluate(tmp_path / "nope")
