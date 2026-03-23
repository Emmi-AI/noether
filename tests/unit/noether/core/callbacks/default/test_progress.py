#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from noether.core.callbacks.default.progress import ProgressCallback


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
    every_n_epochs: int | None = None,
    every_n_updates: int | None = None,
    every_n_samples: int | None = None,
    total_training_updates: int = 100,
) -> ProgressCallback:
    cfg = SimpleNamespace(
        every_n_epochs=every_n_epochs,
        every_n_updates=every_n_updates,
        every_n_samples=every_n_samples,
        batch_size=None,
    )
    end_checkpoint = SimpleNamespace(epoch=10, sample=1000)
    trainer = SimpleNamespace(total_training_updates=total_training_updates, end_checkpoint=end_checkpoint)
    return ProgressCallback(
        callback_config=cfg,
        trainer=trainer,
        model=SimpleNamespace(),
        data_container=SimpleNamespace(),
        tracker=SimpleNamespace(),
        log_writer=SimpleNamespace(),
        checkpoint_writer=SimpleNamespace(),
        metric_property_provider=SimpleNamespace(),
        name=None,
    )


class TestBeforeTraining:
    def test_sets_start_and_last_log_time(self) -> None:
        cb = _make_callback(every_n_updates=10)
        cb.before_training()
        assert cb._start_time is not None
        assert cb._last_log_time is not None
        assert cb._last_track_time is not None


class TestPeriodicCallback:
    def _make_update_counter(
        self,
        *,
        epoch: int = 1,
        update: int = 10,
        sample: int = 100,
        effective_batch_size: int = 10,
    ) -> _UpdateCounter:
        return _UpdateCounter(
            cur_iteration=_Iteration(epoch=epoch, update=update, sample=sample),
            end_iteration=_Iteration(epoch=10, update=100, sample=1000),
            updates_per_epoch=10,
            effective_batch_size=effective_batch_size,
            is_full_epoch=False,
        )

    def test_logs_epoch_progress(self, caplog: pytest.LogCaptureFixture) -> None:
        cb = _make_callback(every_n_epochs=1, total_training_updates=100)
        cb.before_training()
        uc = self._make_update_counter(epoch=2, update=20, sample=200, effective_batch_size=10)

        with caplog.at_level("INFO"):
            cb.periodic_callback(interval_type="epoch", update_counter=uc)

        assert any("Epoch 2/" in msg for msg in caplog.messages)

    def test_logs_update_progress(self, caplog: pytest.LogCaptureFixture) -> None:
        cb = _make_callback(every_n_updates=10, total_training_updates=100)
        cb.before_training()
        uc = self._make_update_counter(update=20, sample=200, effective_batch_size=10)

        with caplog.at_level("INFO"):
            cb.periodic_callback(interval_type="update", update_counter=uc)

        assert any("Update 20/" in msg for msg in caplog.messages)

    def test_logs_sample_progress(self, caplog: pytest.LogCaptureFixture) -> None:
        cb = _make_callback(every_n_samples=100, total_training_updates=100)
        cb.before_training()
        uc = self._make_update_counter(sample=200, effective_batch_size=10)

        with caplog.at_level("INFO"):
            cb.periodic_callback(interval_type="sample", update_counter=uc)

        assert any("Sample 200/" in msg for msg in caplog.messages)

    def test_first_call_resets_start_time(self) -> None:
        cb = _make_callback(every_n_updates=10, total_training_updates=100)
        cb.before_training()
        original_start = cb._start_time
        uc = self._make_update_counter(update=10, sample=100, effective_batch_size=10)

        cb.periodic_callback(interval_type="update", update_counter=uc)

        # _last_log_samples was 0 on first call, so _start_time gets reset
        assert cb._start_time != original_start

    def test_updates_last_log_samples(self) -> None:
        cb = _make_callback(every_n_updates=10, total_training_updates=100)
        cb.before_training()
        assert cb._last_log_samples == 0

        uc = self._make_update_counter(update=10, sample=100, effective_batch_size=10)
        cb.periodic_callback(interval_type="update", update_counter=uc)

        assert cb._last_log_samples == 100

    def test_not_enough_new_samples_since_last_log(self) -> None:
        cb = _make_callback(every_n_updates=1, total_training_updates=100)
        cb.before_training()

        # First call to set _last_log_samples
        uc1 = self._make_update_counter(update=10, sample=100, effective_batch_size=10)
        cb.periodic_callback(interval_type="sample", update_counter=uc1)
        assert cb._last_log_samples == 100

        # Second call where sample advanced by less than effective_batch_size
        # samples_since_last_log = 105 - 100 = 5
        # updates_since_last_log = 5 // 10 = 0  -> ZeroDivisionError
        uc2 = self._make_update_counter(update=11, sample=105, effective_batch_size=10)

        cb.periodic_callback(interval_type="sample", update_counter=uc2)

    def test_no_new_updates_since_last_log(self) -> None:
        cb = _make_callback(every_n_updates=1, total_training_updates=100)
        cb.before_training()

        uc1 = self._make_update_counter(update=10, sample=100, effective_batch_size=10)
        cb.periodic_callback(interval_type="update", update_counter=uc1)

        # Same sample count as last log
        uc2 = self._make_update_counter(update=11, sample=100, effective_batch_size=10)

        cb.periodic_callback(interval_type="update", update_counter=uc2)

    def test_second_call_computes_eta_from_progress_delta(self, caplog: pytest.LogCaptureFixture) -> None:
        cb = _make_callback(every_n_updates=10, total_training_updates=100)
        cb.before_training()

        uc1 = self._make_update_counter(update=10, sample=100, effective_batch_size=10)
        cb.periodic_callback(interval_type="update", update_counter=uc1)

        uc2 = self._make_update_counter(update=20, sample=200, effective_batch_size=10)
        with caplog.at_level("INFO"):
            cb.periodic_callback(interval_type="update", update_counter=uc2)

        assert any("Estimated end time" in msg for msg in caplog.messages)
        assert any("time_per_update" in msg for msg in caplog.messages)


class TestTrackAfterUpdateStep:
    def test_noop_before_training_called(self) -> None:
        """track_after_update_step returns early if _last_track_time or _last_log_time is None."""
        cb = _make_callback(every_n_updates=10, total_training_updates=100)
        # Don't call before_training => _last_track_time is None
        uc = _UpdateCounter(
            cur_iteration=_Iteration(epoch=1, update=1, sample=10),
            end_iteration=_Iteration(epoch=10, update=100, sample=1000),
            updates_per_epoch=10,
            effective_batch_size=10,
            is_full_epoch=False,
        )
        # Should not raise
        cb.track_after_update_step(update_counter=uc)

    def test_does_not_invoke_periodic_if_below_interval(self) -> None:
        cb = _make_callback(every_n_updates=10, total_training_updates=100)
        cb.before_training()
        # Set last log time to now so time_since_last_log < _MIN_LOG_INTERVAL_S
        uc = _UpdateCounter(
            cur_iteration=_Iteration(epoch=1, update=1, sample=10),
            end_iteration=_Iteration(epoch=10, update=100, sample=1000),
            updates_per_epoch=10,
            effective_batch_size=10,
            is_full_epoch=False,
        )
        # Should not invoke periodic_callback because time thresholds not met
        cb.track_after_update_step(update_counter=uc)

    def test_invokes_periodic_when_time_threshold_exceeded(self, caplog: pytest.LogCaptureFixture) -> None:
        cb = _make_callback(every_n_updates=10, total_training_updates=100)
        cb.before_training()

        # Push last log and track times back beyond _MIN_LOG_INTERVAL_S
        past = datetime.now() - timedelta(seconds=cb._MIN_LOG_INTERVAL_S + 10)
        cb._last_log_time = past
        cb._last_track_time = past

        uc = _UpdateCounter(
            cur_iteration=_Iteration(epoch=1, update=10, sample=100),
            end_iteration=_Iteration(epoch=10, update=100, sample=1000),
            updates_per_epoch=10,
            effective_batch_size=10,
            is_full_epoch=False,
        )

        with caplog.at_level("INFO"):
            cb.track_after_update_step(update_counter=uc)

        assert any("Update 10/" in msg for msg in caplog.messages)
