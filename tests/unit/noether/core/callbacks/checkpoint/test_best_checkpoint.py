#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pytest

from noether.core.callbacks.checkpoint.best_checkpoint import BestCheckpointCallback


@pytest.fixture
def callback_deps():
    return {
        "trainer": Mock(),
        "model": Mock(),
        "data_container": Mock(),
        "tracker": Mock(),
        "log_writer": Mock(log_cache={}),
        "checkpoint_writer": Mock(save=Mock()),
        "metric_property_provider": Mock(),
    }


@pytest.fixture
def base_config():
    """Base configuration for BestCheckpointCallback."""
    return {
        "every_n_epochs": None,
        "every_n_updates": 1,
        "every_n_samples": None,
        "batch_size": None,
        "metric_key": "val/acc",
        "model_names": [],
        "model_name": None,
        "tolerances": None,
        "save_frozen_weights": False,
    }


class TestBestCheckpointCallback:
    @pytest.mark.parametrize(
        ("higher_is_better", "current", "new", "expected"),
        [
            (True, 0.5, 0.6, True),
            (True, 0.5, 0.4, False),
            (False, 0.5, 0.4, True),
            (False, 0.5, 0.6, False),
            (True, -float("inf"), 0.1, True),
            (False, float("inf"), 0.1, True),
        ],
    )
    def test_is_new_best_model(self, callback_deps, base_config, higher_is_better, current, new, expected):
        callback_deps["metric_property_provider"].higher_is_better.return_value = higher_is_better
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        cb.best_metric_value = current
        assert cb._is_new_best_model(new) == expected

    def test_saves_best_and_tolerance_checkpoints(self, callback_deps, base_config):
        callback_deps["metric_property_provider"].higher_is_better.return_value = True
        base_config["tolerances"] = [1, 3]
        callback_deps["log_writer"].log_cache = {"val/acc": 0.9}

        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        cb.periodic_callback()

        assert callback_deps["checkpoint_writer"].save.call_count == 3
        tags = [call.kwargs["checkpoint_tag"] for call in callback_deps["checkpoint_writer"].save.call_args_list]
        assert "best_model.val.acc" in tags

    def test_tolerance_counter_increments_and_triggers_save(self, callback_deps, base_config):
        """
        Tests that tolerance behaves like a 'patience' counter.
        Config: Tolerance = 2.
        Expectation:
          - Fail 1 (Counter 1): OK (1 <= 2)
          - Fail 2 (Counter 2): OK (2 <= 2)
          - Fail 3 (Counter 3): Exceeded (3 > 2)
        """
        callback_deps["metric_property_provider"].higher_is_better.return_value = True
        base_config["tolerances"] = [2]

        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        cb.best_metric_value = 0.9

        # 1. First Failure:
        callback_deps["log_writer"].log_cache = {"val/acc": 0.85}
        cb.periodic_callback()
        assert cb.tolerance_counter == 1
        assert cb.tolerances_is_exceeded.get(2, False) is False

        # 2. Second Failure (Tolerance limit reached, but not exceeded):
        cb.periodic_callback()
        assert cb.tolerance_counter == 2
        assert cb.tolerances_is_exceeded.get(2, False) is False

        # 3. Third Failure (Exceeded):
        cb.periodic_callback()
        assert cb.tolerance_counter == 3
        assert cb.tolerances_is_exceeded[2] is True
        assert cb.metric_at_exceeded_tolerance[2] == 0.85

    def test_tolerance_counter_resets_on_new_best(self, callback_deps, base_config):
        """Tolerance counter AND exceeded flags must reset on new best model."""
        callback_deps["metric_property_provider"].higher_is_better.return_value = True
        base_config["tolerances"] = [5]

        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)

        callback_deps["log_writer"].log_cache = {"val/acc": 0.9}
        cb.periodic_callback()

        # Fail a few times:
        callback_deps["log_writer"].log_cache = {"val/acc": 0.85}
        cb.periodic_callback()
        cb.periodic_callback()
        assert cb.tolerance_counter == 2

        # New best:
        callback_deps["log_writer"].log_cache = {"val/acc": 0.95}
        cb.periodic_callback()

        assert cb.tolerance_counter == 0
        # Important: Ensure the state dict for exceeded is also reset:
        assert all(v is False for v in cb.tolerances_is_exceeded.values())

    def test_state_dict_round_trip(self, callback_deps, base_config):
        callback_deps["metric_property_provider"].higher_is_better.return_value = True
        base_config["tolerances"] = [1]
        callback_deps["log_writer"].log_cache = {"val/acc": 0.9}

        cb1 = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        cb1.periodic_callback()

        callback_deps["log_writer"].log_cache = {"val/acc": 0.85}
        cb1.periodic_callback()  # Counter 1
        cb1.periodic_callback()  # Counter 2 (Exceeded for tolerance 1)

        state = cb1.state_dict()
        cb2 = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        cb2.load_state_dict(state)

        assert cb2.best_metric_value == 0.9
        assert cb2.tolerance_counter == 2
        assert cb2.tolerances_is_exceeded == {1: True}

    def test_raises_on_missing_log_cache(self, callback_deps, base_config):
        callback_deps["log_writer"].log_cache = None
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        with pytest.raises(KeyError, match="Log cache is empty"):
            cb.periodic_callback()

    def test_raises_on_missing_metric_key(self, callback_deps, base_config):
        callback_deps["log_writer"].log_cache = {"other": 0.5}
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        with pytest.raises(KeyError, match="couldn't find metric_key"):
            cb.periodic_callback()

    def test_after_training_logs_tolerance_metrics(self, callback_deps, base_config):
        callback_deps["metric_property_provider"].higher_is_better.return_value = True
        base_config["tolerances"] = [1]

        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)

        with patch(
            "noether.core.callbacks.checkpoint.best_checkpoint.BestCheckpointCallback.logger", new_callable=PropertyMock
        ) as mock_logger:
            mock_log_instance = Mock()
            mock_logger.return_value = mock_log_instance

            callback_deps["log_writer"].log_cache = {"val/acc": 0.9}
            cb.periodic_callback()

            callback_deps["log_writer"].log_cache = {"val/acc": 0.85}
            cb.periodic_callback()  # Counter 1
            cb.periodic_callback()  # Counter 2 (Exceeds 1)

            cb.after_training()

            log_calls = [call[0][0] for call in mock_log_instance.info.call_args_list]
            assert any("tolerance=1" in log for log in log_calls)
