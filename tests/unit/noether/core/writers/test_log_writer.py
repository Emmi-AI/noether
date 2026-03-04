#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from noether.core.providers import PathProvider
from noether.core.trackers import BaseTracker
from noether.core.utils.training import TrainingIteration, UpdateCounter
from noether.core.writers.log_writer import LogWriter

_MODULE_PATH = "noether.core.writers.log_writer"


def _make_update_counter(epoch=1, update=10, sample=40) -> UpdateCounter:
    start = TrainingIteration(epoch=0, update=0, sample=0)
    end = TrainingIteration(epoch=5, update=None, sample=None)
    uc = UpdateCounter(start_iteration=start, end_iteration=end, updates_per_epoch=10, effective_batch_size=4)
    uc.cur_iteration = TrainingIteration(epoch=epoch, update=update, sample=sample)
    return uc


def _make_log_writer(tmp_path: Path, epoch=1, update=10, sample=40) -> LogWriter:
    path_provider = PathProvider(output_root_path=tmp_path, run_id="test-run")
    update_counter = _make_update_counter(epoch=epoch, update=update, sample=sample)
    tracker = MagicMock(spec=BaseTracker)
    return LogWriter(path_provider=path_provider, update_counter=update_counter, tracker=tracker)


class TestLogWriterInit:
    def test_initial_state(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        assert lw.log_entries == []
        assert lw.log_cache is None
        assert lw.non_scalar_keys == set()


class TestAddScalar:
    def test_scalar_float(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        lw.add_scalar("loss", 0.5)
        assert lw.log_cache["loss"] == 0.5

    def test_scalar_tensor_converted_to_python(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        lw.add_scalar("loss", torch.tensor(1.5))
        assert lw.log_cache["loss"] == 1.5
        assert isinstance(lw.log_cache["loss"], float)

    def test_scalar_numpy_converted_to_python(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        lw.add_scalar("loss", np.float32(2.5))
        assert lw.log_cache["loss"] == pytest.approx(2.5)
        assert not isinstance(lw.log_cache["loss"], np.generic)

    def test_first_scalar_initializes_cache_with_counters(self, tmp_path):
        lw = _make_log_writer(tmp_path, epoch=3, update=30, sample=120)
        lw.add_scalar("lr", 0.01)
        assert lw.log_cache["epoch"] == 3
        assert lw.log_cache["update"] == 30
        assert lw.log_cache["sample"] == 120

    def test_duplicate_key_raises(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        lw.add_scalar("loss", 1.0)
        with pytest.raises(KeyError, match="already logged"):
            lw.add_scalar("loss", 2.0)

    def test_trailing_slash_stripped(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        lw.add_scalar("loss/", 1.0)
        assert "loss" in lw.log_cache
        assert "loss/" not in lw.log_cache

    def test_scalar_logged_to_logger(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        mock_logger = MagicMock(spec=logging.Logger)
        lw.add_scalar("loss", 1.234, logger=mock_logger)
        mock_logger.info.assert_called_once()
        assert "loss" in mock_logger.info.call_args[0][0]

    def test_scalar_logged_with_format_str(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        mock_logger = MagicMock(spec=logging.Logger)
        lw.add_scalar("loss", 1.23456789, logger=mock_logger, format_str=".2f")
        assert "1.23" in mock_logger.info.call_args[0][0]

    def test_scalar_not_added_to_non_scalar_keys(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        lw.add_scalar("loss", 1.0)
        assert "loss" not in lw.non_scalar_keys


class TestAddNonScalar:
    def test_nonscalar_added_to_cache(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        lw.add_nonscalar("histogram", [1, 2, 3])
        assert lw.log_cache["histogram"] == [1, 2, 3]

    def test_nonscalar_key_tracked(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        lw.add_nonscalar("image", "img_data")
        assert "image" in lw.non_scalar_keys


class TestFlush:
    def test_flush_sends_cache_to_tracker(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        lw.add_scalar("loss", 1.0)
        cache_copy = dict(lw.log_cache)
        lw.flush()
        lw.tracker.log.assert_called_once_with(cache_copy)

    def test_flush_appends_scalar_entries_to_log(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        lw.add_scalar("loss", 1.0)
        lw.flush()
        assert len(lw.log_entries) == 1
        assert lw.log_entries[0]["loss"] == 1.0

    def test_flush_clears_cache(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        lw.add_scalar("loss", 1.0)
        lw.flush()
        assert lw.log_cache is None

    def test_flush_noop_when_cache_empty(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        lw.flush()
        lw.tracker.log.assert_not_called()
        assert len(lw.log_entries) == 0

    def test_flush_filters_nonscalar_from_entries(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        lw.add_scalar("loss", 1.0)
        lw.add_nonscalar("image", "img_data")
        lw.flush()
        assert "image" not in lw.log_entries[0]
        assert "loss" in lw.log_entries[0]

    def test_flush_clears_nonscalar_keys(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        lw.add_nonscalar("img", "data")
        lw.flush()
        assert lw.non_scalar_keys == set()

    def test_flush_enforces_monotonic_updates(self, tmp_path):
        lw = _make_log_writer(tmp_path, update=10)
        lw.add_scalar("loss", 1.0)
        lw.flush()

        # Second flush at same update should fail:
        lw.update_counter.cur_iteration = TrainingIteration(epoch=1, update=10, sample=40)
        lw.add_scalar("loss", 2.0)
        with pytest.raises(AssertionError):
            lw.flush()

    def test_multiple_flushes_accumulate_entries(self, tmp_path):
        lw = _make_log_writer(tmp_path, update=10)
        lw.add_scalar("loss", 1.0)
        lw.flush()

        lw.update_counter.cur_iteration = TrainingIteration(epoch=1, update=20, sample=80)
        lw.add_scalar("loss", 0.5)
        lw.flush()

        assert len(lw.log_entries) == 2
        assert lw.log_entries[0]["loss"] == 1.0
        assert lw.log_entries[1]["loss"] == 0.5


class TestGetAllMetricValues:
    def test_returns_values_across_entries(self, tmp_path):
        lw = _make_log_writer(tmp_path, update=10)
        lw.add_scalar("loss", 1.0)
        lw.flush()

        lw.update_counter.cur_iteration = TrainingIteration(epoch=1, update=20, sample=80)
        lw.add_scalar("loss", 0.5)
        lw.flush()

        values = lw.get_all_metric_values("loss")
        assert values == [1.0, 0.5]

    def test_skips_entries_without_key(self, tmp_path):
        lw = _make_log_writer(tmp_path, update=10)
        lw.add_scalar("loss", 1.0)
        lw.flush()

        lw.update_counter.cur_iteration = TrainingIteration(epoch=1, update=20, sample=80)
        lw.add_scalar("lr", 0.01)
        lw.flush()

        values = lw.get_all_metric_values("loss")
        assert values == [1.0]

    def test_empty_when_no_entries(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        assert lw.get_all_metric_values("loss") == []


class TestFinish:
    def test_finish_saves_entries_to_disk(self, tmp_path):
        lw = _make_log_writer(tmp_path, update=10)
        lw.add_scalar("loss", 1.0)
        lw.add_scalar("lr", 0.01)
        lw.flush()

        lw.update_counter.cur_iteration = TrainingIteration(epoch=1, update=20, sample=80)
        lw.add_scalar("loss", 0.5)
        lw.flush()

        with patch(_MODULE_PATH + ".is_rank0", return_value=True):
            lw.finish()

        entries_path = lw.path_provider.basetracker_entries_uri
        assert entries_path.exists()
        data = torch.load(entries_path, weights_only=False)
        assert "loss" in data
        assert data["loss"][10] == 1.0
        assert data["loss"][20] == 0.5

    def test_finish_noop_when_no_entries(self, tmp_path):
        lw = _make_log_writer(tmp_path)
        with patch(_MODULE_PATH + ".is_rank0", return_value=True):
            lw.finish()
        assert not lw.path_provider.basetracker_entries_uri.exists()

    def test_finish_noop_when_not_rank0(self, tmp_path):
        lw = _make_log_writer(tmp_path, update=10)
        lw.add_scalar("loss", 1.0)
        lw.flush()

        with patch(_MODULE_PATH + ".is_rank0", return_value=False):
            lw.finish()

        assert not lw.path_provider.basetracker_entries_uri.exists()

    def test_finish_groups_by_key_and_update(self, tmp_path):
        lw = _make_log_writer(tmp_path, update=10)
        lw.add_scalar("loss", 1.0)
        lw.add_scalar("acc", 0.8)
        lw.flush()

        with patch(_MODULE_PATH + ".is_rank0", return_value=True):
            lw.finish()

        data = torch.load(lw.path_provider.basetracker_entries_uri, weights_only=False)
        assert data["loss"] == {10: 1.0}
        assert data["acc"] == {10: 0.8}

    def test_finish_excludes_update_key_from_result(self, tmp_path):
        lw = _make_log_writer(tmp_path, update=10)
        lw.add_scalar("loss", 1.0)
        lw.flush()

        with patch(_MODULE_PATH + ".is_rank0", return_value=True):
            lw.finish()

        data = torch.load(lw.path_provider.basetracker_entries_uri, weights_only=False)
        # "update" key should not appear as a top-level metric:
        assert "update" not in data
