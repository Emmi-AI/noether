#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from noether.core.callbacks.default.eta import EtaCallback

_MONKEY_PATCH_IS_RANK0 = "noether.core.callbacks.default.eta.is_rank0"


@dataclass(frozen=True)
class _Iteration:
    epoch: int | None
    update: int | None
    sample: int | None


@dataclass
class _UpdateCounter:
    cur_iteration: _Iteration
    end_iteration: _Iteration
    updates_per_epoch: int
    effective_batch_size: int
    is_full_epoch: bool


def _make_callback(
    *,
    every_n_epochs: int | None,
    every_n_updates: int | None,
    every_n_samples: int | None,
) -> EtaCallback:
    cfg = SimpleNamespace(
        every_n_epochs=every_n_epochs,
        every_n_updates=every_n_updates,
        every_n_samples=every_n_samples,
        batch_size=None,
    )
    return EtaCallback(
        callback_config=cfg,  # runtime duck-typed
        trainer=SimpleNamespace(),  # not used by EtaCallback in these tests
        model=SimpleNamespace(),
        data_container=SimpleNamespace(),
        tracker=SimpleNamespace(),
        log_writer=SimpleNamespace(),
        checkpoint_writer=SimpleNamespace(),
        metric_property_provider=SimpleNamespace(),
        name=None,
    )


def test_before_training_requires_rank0(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_MONKEY_PATCH_IS_RANK0, lambda: False)

    cb = _make_callback(every_n_epochs=1, every_n_updates=None, every_n_samples=None)
    uc = _UpdateCounter(
        cur_iteration=_Iteration(epoch=1, update=1, sample=2),
        end_iteration=_Iteration(epoch=10, update=100, sample=200),
        updates_per_epoch=10,
        effective_batch_size=2,
        is_full_epoch=False,
    )

    with pytest.raises(AssertionError, match="only use EtaCallback on rank0 process"):
        cb.before_training(update_counter=uc)


@pytest.mark.parametrize(
    ("end_epoch", "end_update", "expected_epoch_fmt", "expected_update_fmt"),
    [
        (1, 1, "1d", "1d"),
        (9, 9, "1d", "1d"),
        (10, 10, "2d", "2d"),
        (12, 120, "2d", "3d"),
    ],
)
def test_before_training_sets_formats_for_epoch_and_update(
    monkeypatch: pytest.MonkeyPatch,
    end_epoch: int,
    end_update: int,
    expected_epoch_fmt: str,
    expected_update_fmt: str,
) -> None:
    monkeypatch.setattr(_MONKEY_PATCH_IS_RANK0, lambda: True)

    cb = _make_callback(every_n_epochs=2, every_n_updates=None, every_n_samples=None)
    uc = _UpdateCounter(
        cur_iteration=_Iteration(epoch=1, update=1, sample=1),
        end_iteration=_Iteration(epoch=end_epoch, update=end_update, sample=200),
        updates_per_epoch=7,
        effective_batch_size=4,
        is_full_epoch=False,
    )

    cb.before_training(update_counter=uc)

    assert cb.epoch_format == expected_epoch_fmt
    assert cb.update_format == expected_update_fmt
    assert cb.every_n_epochs_format is not None
    assert cb.updates_per_log_interval_format is not None
    assert cb._start_time is not None


def test_before_training_sets_sample_schedule_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_MONKEY_PATCH_IS_RANK0, lambda: True)

    cb = _make_callback(every_n_epochs=None, every_n_updates=None, every_n_samples=5)
    uc = _UpdateCounter(
        cur_iteration=_Iteration(epoch=1, update=1, sample=1),
        end_iteration=_Iteration(epoch=10, update=100, sample=200),
        updates_per_epoch=10,
        effective_batch_size=2,
        is_full_epoch=False,
    )

    cb.before_training(update_counter=uc)

    # ceil(5 / 2) = 3 updates per log interval at the start
    assert hasattr(cb, "updates_per_every_n_samples")
    assert cb.updates_per_every_n_samples == np.ceil(5 / 2)
    assert cb.updates_per_log_interval_format is not None


def test_track_after_update_step_prints_carriage_return_by_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(_MONKEY_PATCH_IS_RANK0, lambda: True)

    cb = _make_callback(every_n_epochs=None, every_n_updates=2, every_n_samples=None)
    uc = _UpdateCounter(
        cur_iteration=_Iteration(epoch=1, update=1, sample=2),
        end_iteration=_Iteration(epoch=5, update=10, sample=20),
        updates_per_epoch=2,
        effective_batch_size=2,
        is_full_epoch=False,
    )
    cb.before_training(update_counter=uc)

    cb.track_after_update_step(update_counter=uc, times={"data_wait": 0.2, "update": 0.8})
    err = capsys.readouterr().err

    assert "E " in err
    assert "U " in err
    assert "S " in err
    assert "avg_update" in err
    assert err.endswith("\r")  # end="\r" path


def test_track_after_update_step_prints_newline_when_logger_was_called(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(_MONKEY_PATCH_IS_RANK0, lambda: True)

    cb = _make_callback(every_n_epochs=None, every_n_updates=2, every_n_samples=None)
    uc = _UpdateCounter(
        cur_iteration=_Iteration(epoch=1, update=1, sample=2),
        end_iteration=_Iteration(epoch=5, update=10, sample=20),
        updates_per_epoch=2,
        effective_batch_size=2,
        is_full_epoch=False,
    )
    cb.before_training(update_counter=uc)

    cb.handler.was_called = True
    cb.track_after_update_step(update_counter=uc, times={"data_wait": 0.1, "update": 0.1})
    err = capsys.readouterr().err

    assert err.endswith("\n")  # print(..., file=stderr) default newline
    assert cb.handler.was_called is False


def test_time_since_last_log_resets_on_interval_boundary_update_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_MONKEY_PATCH_IS_RANK0, lambda: True)

    cb = _make_callback(every_n_epochs=None, every_n_updates=2, every_n_samples=None)
    uc = _UpdateCounter(
        cur_iteration=_Iteration(epoch=1, update=2, sample=4),  # boundary: 2 % 2 == 0
        end_iteration=_Iteration(epoch=5, update=10, sample=20),
        updates_per_epoch=2,
        effective_batch_size=2,
        is_full_epoch=False,
    )
    cb.before_training(update_counter=uc)

    cb.time_since_last_log = 123.0
    cb.track_after_update_step(update_counter=uc, times={"data_wait": 0.25, "update": 0.75})

    # reset to 0.0 then add increment => 1.0
    assert cb.time_since_last_log == pytest.approx(1.0)


def test_periodic_callback_prints_newline_only_for_update(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cb = _make_callback(every_n_epochs=None, every_n_updates=1, every_n_samples=None)

    cb.periodic_callback(interval_type="epoch")
    assert capsys.readouterr().err == ""

    cb.periodic_callback(interval_type="update")
    assert capsys.readouterr().err == "\n"


def test_after_training_removes_handler_from_root_logger() -> None:
    cb = _make_callback(every_n_epochs=1, every_n_updates=None, every_n_samples=None)

    root = logging.getLogger()
    root.addHandler(cb.handler)
    try:
        cb.after_training()
        assert cb.handler not in root.handlers
    finally:
        if cb.handler in root.handlers:
            root.removeHandler(cb.handler)


def test_epoch_boundary_resets_time_since_last_log(monkeypatch: pytest.MonkeyPatch) -> None:
    # Covers:
    # if self._should_log_after_epoch(...) and update_counter.is_full_epoch: self.time_since_last_log = 0.0
    monkeypatch.setattr("noether.core.callbacks.default.eta.is_rank0", lambda: True)

    cb = _make_callback(every_n_epochs=2, every_n_updates=None, every_n_samples=None)
    uc = _UpdateCounter(
        cur_iteration=_Iteration(epoch=2, update=20, sample=40),  # epoch boundary for every_n_epochs=2
        end_iteration=_Iteration(epoch=10, update=100, sample=200),
        updates_per_epoch=10,
        effective_batch_size=2,
        is_full_epoch=True,  # required to hit the epoch reset line
    )
    cb.before_training(update_counter=uc)

    cb.time_since_last_log = 123.0
    cb.track_after_update_step(update_counter=uc, times={"data_wait": 0.25, "update": 0.75})

    # reset to 0.0 then +1.0
    assert cb.time_since_last_log == pytest.approx(1.0)


def test_epoch_branch_updates_since_last_log_zero_becomes_full_interval(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Covers:
    # if self.every_n_epochs: ... if updates_since_last_log == 0: updates_since_last_log = updates_per_log_interval
    # We detect it via "next_log X/X" in stderr.
    monkeypatch.setattr("noether.core.callbacks.default.eta.is_rank0", lambda: True)

    cb = _make_callback(every_n_epochs=2, every_n_updates=None, every_n_samples=None)

    # For every_n_epochs=2, updates_per_epoch=10:
    # at epoch=2 => last_epoch=2, updates_at_last_log=20
    # set cur_update=20 => updates_since_last_log=0 => branch sets it to updates_per_log_interval (=20)
    uc = _UpdateCounter(
        cur_iteration=_Iteration(epoch=2, update=20, sample=40),
        end_iteration=_Iteration(epoch=10, update=100, sample=200),
        updates_per_epoch=10,
        effective_batch_size=2,
        is_full_epoch=True,
    )
    cb.before_training(update_counter=uc)

    cb.track_after_update_step(update_counter=uc, times={"data_wait": 0.0, "update": 1.0})
    err = capsys.readouterr().err

    # Remove whitespace to avoid format padding issues.
    compact = "".join(err.split())
    assert "next_log20/20" in compact  # proves the "== 0" branch executed


def test_sample_branch_computes_log_interval_with_superfluous_samples(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Covers the entire `elif self.every_n_samples:` block (lines ~81-88 in your screenshot).
    monkeypatch.setattr("noether.core.callbacks.default.eta.is_rank0", lambda: True)

    cb = _make_callback(every_n_epochs=None, every_n_updates=None, every_n_samples=5)

    # Choose effective_batch_size=4 and sample=9:
    # samples_since_last_log = 9 % 5 = 4
    # samples_at_last_log = 9 - 4 = 5
    # updates_at_last_log = 5 // 4 = 1
    # superfluous_samples_at_last_log = 5 % 4 = 1
    # samples_for_cur_log_interval = 5 - 1 = 4
    # updates_per_log_interval = ceil(4 / 4) = 1
    # Use cur_update=3 => updates_since_last_log = 3 - 1 = 2
    uc = _UpdateCounter(
        cur_iteration=_Iteration(epoch=1, update=3, sample=9),
        end_iteration=_Iteration(epoch=10, update=100, sample=200),
        updates_per_epoch=10,
        effective_batch_size=4,
        is_full_epoch=False,
    )
    cb.before_training(update_counter=uc)

    cb.track_after_update_step(update_counter=uc, times={"data_wait": 0.0, "update": 1.0})
    err = capsys.readouterr().err
    compact = "".join(err.split())

    # next_log <updates_since_last_log>/<updates_per_log_interval>
    assert "next_log2/1" in compact


def test_sample_boundary_resets_time_since_last_log(monkeypatch: pytest.MonkeyPatch) -> None:
    # Covers:
    # if self._should_log_after_sample(...): self.time_since_last_log = 0.0
    monkeypatch.setattr("noether.core.callbacks.default.eta.is_rank0", lambda: True)

    cb = _make_callback(every_n_epochs=None, every_n_updates=None, every_n_samples=5)

    # Make _should_log_after_sample True:
    # effective_batch_size=2, sample=6 => last_update_samples=4
    # prev_log_step=int(4/5)=0, cur_log_step=int(6/5)=1 => True
    uc = _UpdateCounter(
        cur_iteration=_Iteration(epoch=1, update=3, sample=6),
        end_iteration=_Iteration(epoch=10, update=100, sample=200),
        updates_per_epoch=10,
        effective_batch_size=2,
        is_full_epoch=False,
    )
    cb.before_training(update_counter=uc)

    cb.time_since_last_log = 77.0
    cb.track_after_update_step(update_counter=uc, times={"data_wait": 0.25, "update": 0.75})

    assert cb.time_since_last_log == pytest.approx(1.0)
