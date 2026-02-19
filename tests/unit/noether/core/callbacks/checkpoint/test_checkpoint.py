#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pytest

from noether.core.callbacks.checkpoint import CheckpointCallback
from noether.core.schemas.callbacks import CheckpointCallbackConfig


@pytest.fixture
def callback_deps():
    return {
        "trainer": Mock(),
        "model": Mock(frozen_param_count=1_000_000, trainable_param_count=2_000_000),
        "data_container": Mock(),
        "tracker": Mock(),
        "log_writer": Mock(),
        "checkpoint_writer": Mock(save=Mock()),
        "metric_property_provider": Mock(),
    }


class TestCheckpointCallback:
    def test_requires_at_least_one_save_flag(self, callback_deps):
        # Using constructor to avoid validation errors on missing fields:
        config = CheckpointCallbackConfig(
            every_n_updates=10,
            save_weights=False,
            save_latest_weights=False,
            save_optim=False,
            save_latest_optim=False,
        )
        with pytest.raises(ValueError, match="At least one of save_weights"):
            CheckpointCallback(callback_config=config, **callback_deps)

    @pytest.mark.parametrize("flag", ["save_weights", "save_latest_weights", "save_optim", "save_latest_optim"])
    def test_initialization_succeeds_with_any_save_flag(self, callback_deps, flag):
        config = CheckpointCallbackConfig(every_n_updates=10, **{flag: True})
        callback = CheckpointCallback(callback_config=config, **callback_deps)
        assert getattr(callback, flag) is True

    def test_model_name_added_to_list(self, callback_deps):
        config = CheckpointCallbackConfig(every_n_updates=10, model_names=["encoder"])
        callback = CheckpointCallback(callback_config=config, **callback_deps)
        assert callback.model_names == ["encoder"]

    @pytest.mark.parametrize(
        ("interval_config", "end_iteration"),
        [
            ({"every_n_epochs": 5}, {"epoch": 20, "update": None, "sample": None}),
            ({"every_n_updates": 100}, {"epoch": None, "update": 1000, "sample": None}),
            # added batch_size here to satisfy Pydantic validation for sample-based config:
            ({"every_n_samples": 10000, "batch_size": 32}, {"epoch": None, "update": None, "sample": 100000}),
        ],
    )
    def test_before_training_logs_size_estimation(self, callback_deps, interval_config, end_iteration):
        config = CheckpointCallbackConfig(save_weights=True, **interval_config)
        callback = CheckpointCallback(callback_config=config, **callback_deps)
        update_counter = SimpleNamespace(end_iteration=SimpleNamespace(**end_iteration))

        with patch(
            "noether.core.callbacks.checkpoint.checkpoint.CheckpointCallback.logger", new_callable=PropertyMock
        ) as mock_logger:
            mock_log_instance = Mock()
            mock_logger.return_value = mock_log_instance

            callback.before_training(update_counter=update_counter)
            mock_log_instance.info.assert_called_once()
            assert "Estimated sizes:" in mock_log_instance.info.call_args[0][0]

    def test_before_training_logs_latest_only_message(self, callback_deps):
        config = CheckpointCallbackConfig(
            every_n_updates=10, save_latest_weights=True, save_weights=False, save_optim=False
        )
        callback = CheckpointCallback(callback_config=config, **callback_deps)
        update_counter = SimpleNamespace(end_iteration=SimpleNamespace(epoch=None, update=100, sample=None))

        with patch(
            "noether.core.callbacks.checkpoint.checkpoint.CheckpointCallback.logger", new_callable=PropertyMock
        ) as mock_logger:
            mock_log_instance = Mock()
            mock_logger.return_value = mock_log_instance

            callback.before_training(update_counter=update_counter)
            assert "only latest weights/optim saved" in mock_log_instance.info.call_args[0][0]

    def test_periodic_callback_skips_eval_interval(self, callback_deps):
        config = CheckpointCallbackConfig(every_n_updates=10)
        callback = CheckpointCallback(callback_config=config, **callback_deps)
        callback.periodic_callback(interval_type="eval", update_counter=SimpleNamespace(cur_iteration="u=10"))
        callback_deps["checkpoint_writer"].save.assert_not_called()

    @pytest.mark.parametrize(
        "save_flags",
        [
            {"save_weights": True, "save_optim": False},
            {"save_weights": False, "save_optim": True},
            {"save_weights": True, "save_optim": True, "save_latest_weights": True},
        ],
    )
    def test_periodic_callback_saves_with_correct_flags(self, callback_deps, save_flags):
        config = CheckpointCallbackConfig(every_n_updates=10, **save_flags)
        callback = CheckpointCallback(callback_config=config, **callback_deps)
        callback.periodic_callback(interval_type="update", update_counter=SimpleNamespace(cur_iteration="u=10"))

        call_kwargs = callback_deps["checkpoint_writer"].save.call_args.kwargs
        assert call_kwargs["checkpoint_tag"] == "u=10"
        for flag, value in save_flags.items():
            assert call_kwargs[flag] == value

    def test_after_training_saves_last_checkpoint(self, callback_deps):
        config = CheckpointCallbackConfig(every_n_updates=10, save_weights=True)
        callback = CheckpointCallback(callback_config=config, **callback_deps)
        callback.after_training()
        call_kwargs = callback_deps["checkpoint_writer"].save.call_args.kwargs
        assert call_kwargs["checkpoint_tag"] == "last"
