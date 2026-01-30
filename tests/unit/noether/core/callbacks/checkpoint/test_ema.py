#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from noether.core.callbacks.checkpoint.ema import EmaCallback

_MODULE_PATH = "noether.core.callbacks.checkpoint.ema"


@pytest.fixture
def callback_deps():
    return {
        "trainer": Mock(),
        "data_container": Mock(),
        "tracker": Mock(),
        "log_writer": Mock(),
        "checkpoint_writer": Mock(save_model_checkpoint=Mock()),
        "metric_property_provider": Mock(),
    }


@pytest.fixture
def base_config():
    return {
        "every_n_epochs": None,
        "every_n_updates": 1,
        "every_n_samples": None,
        "batch_size": None,
        "model_paths": [None],
        "target_factors": [0.9],
        "save_weights": False,
        "save_last_weights": False,
        "save_latest_weights": False,
    }


class _TinyModel(torch.nn.Module):
    def __init__(self, name="tiny"):
        super().__init__()
        self.name = name
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.register_buffer("buf", torch.ones(1))


class _NestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "nested"
        self.encoder = _TinyModel(name="encoder")
        self.decoder = _TinyModel(name="decoder")


class TestEmaCallback:
    def test_before_training_initializes_shadow_params(self, monkeypatch, callback_deps, base_config):
        monkeypatch.setattr(_MODULE_PATH + ".is_rank0", lambda: True)
        monkeypatch.setattr(_MODULE_PATH + ".select_with_path", lambda *, obj, path: obj)

        model = _TinyModel()
        cb = EmaCallback(callback_config=SimpleNamespace(**base_config), model=model, **callback_deps)
        cb.before_training()

        assert (None, 0.9) in cb.parameters
        assert torch.equal(cb.parameters[(None, 0.9)]["linear.weight"], model.linear.weight)
        assert cb.parameters[(None, 0.9)]["linear.weight"].data_ptr() != model.linear.weight.data_ptr()
        assert torch.equal(cb.buffers[None]["buf"], model.buf)

    def test_track_after_update_step_applies_ema(self, monkeypatch, callback_deps, base_config):
        monkeypatch.setattr(_MODULE_PATH + ".is_rank0", lambda: True)
        monkeypatch.setattr(_MODULE_PATH + ".select_with_path", lambda *, obj, path: obj)

        model = _TinyModel()
        base_config["target_factors"] = [0.5]

        with torch.no_grad():
            model.linear.weight.fill_(2.0)

        cb = EmaCallback(callback_config=SimpleNamespace(**base_config), model=model, **callback_deps)
        cb.before_training()

        with torch.no_grad():
            model.linear.weight.fill_(4.0)
            model.buf.fill_(9.0)

        cb.track_after_update_step()

        # EMA: 0.5 * 2.0 + 0.5 * 4.0 = 3.0
        shadow = cb.parameters[(None, 0.5)]["linear.weight"]
        assert torch.allclose(shadow, torch.full_like(shadow, 3.0))
        assert torch.equal(cb.buffers[None]["buf"], model.buf)

    @pytest.mark.parametrize(
        ("save_flag", "count"),
        [
            ("save_weights", 1),
            ("save_latest_weights", 1),
            ("save_last_weights", 0),
        ],
    )
    def test_periodic_callback_respects_save_flags(self, monkeypatch, callback_deps, base_config, save_flag, count):
        monkeypatch.setattr(_MODULE_PATH + ".is_rank0", lambda: True)
        monkeypatch.setattr(_MODULE_PATH + ".select_with_path", lambda *, obj, path: obj)
        monkeypatch.setattr(_MODULE_PATH + ".ModelBase", torch.nn.Module)

        model = _TinyModel()
        base_config[save_flag] = True

        cb = EmaCallback(callback_config=SimpleNamespace(**base_config), model=model, **callback_deps)
        cb.before_training()

        cb.periodic_callback(interval_type="update", update_counter=SimpleNamespace(cur_iteration="u=10"))

        assert callback_deps["checkpoint_writer"].save_model_checkpoint.call_count == count

    def test_periodic_callback_skips_eval_interval(self, monkeypatch, callback_deps, base_config):
        monkeypatch.setattr(_MODULE_PATH + ".is_rank0", lambda: True)
        monkeypatch.setattr(_MODULE_PATH + ".select_with_path", lambda *, obj, path: obj)

        model = _TinyModel()
        base_config["save_weights"] = True

        cb = EmaCallback(callback_config=SimpleNamespace(**base_config), model=model, **callback_deps)
        cb.before_training()
        cb.periodic_callback(interval_type="eval", update_counter=SimpleNamespace(cur_iteration="u=10"))

        callback_deps["checkpoint_writer"].save_model_checkpoint.assert_not_called()

    def test_multiple_target_factors(self, monkeypatch, callback_deps, base_config):
        monkeypatch.setattr(_MODULE_PATH + ".is_rank0", lambda: True)
        monkeypatch.setattr(_MODULE_PATH + ".select_with_path", lambda *, obj, path: obj)

        model = _TinyModel()
        base_config["target_factors"] = [0.5, 0.9]

        with torch.no_grad():
            model.linear.weight.fill_(2.0)

        cb = EmaCallback(callback_config=SimpleNamespace(**base_config), model=model, **callback_deps)
        cb.before_training()

        with torch.no_grad():
            model.linear.weight.fill_(10.0)

        cb.track_after_update_step()

        # Verify different EMA results
        shadow_05 = cb.parameters[(None, 0.5)]["linear.weight"]
        shadow_09 = cb.parameters[(None, 0.9)]["linear.weight"]
        assert torch.allclose(shadow_05, torch.full_like(shadow_05, 6.0))  # 0.5 * 2 + 0.5 * 10
        assert torch.allclose(shadow_09, torch.full_like(shadow_09, 2.8))  # 0.9 * 2 + 0.1 * 10

    def test_multiple_model_paths(self, monkeypatch, callback_deps, base_config):
        monkeypatch.setattr(_MODULE_PATH + ".is_rank0", lambda: True)

        def mock_select(*, obj, path):
            return obj.encoder if path == "encoder" else obj.decoder if path == "decoder" else obj

        monkeypatch.setattr(_MODULE_PATH + ".select_with_path", mock_select)

        model = _NestedModel()
        base_config["model_paths"] = ["encoder", "decoder"]
        base_config["target_factors"] = [0.5]

        with torch.no_grad():
            model.encoder.linear.weight.fill_(2.0)
            model.decoder.linear.weight.fill_(4.0)

        cb = EmaCallback(callback_config=SimpleNamespace(**base_config), model=model, **callback_deps)
        cb.before_training()

        with torch.no_grad():
            model.encoder.linear.weight.fill_(6.0)
            model.decoder.linear.weight.fill_(8.0)

        cb.track_after_update_step()

        # Verify separate EMA for each path:
        encoder_shadow = cb.parameters[("encoder", 0.5)]["linear.weight"]
        decoder_shadow = cb.parameters[("decoder", 0.5)]["linear.weight"]
        assert torch.allclose(encoder_shadow, torch.full_like(encoder_shadow, 4.0))
        assert torch.allclose(decoder_shadow, torch.full_like(decoder_shadow, 6.0))

    def test_resume_from_checkpoint(self, monkeypatch, tmp_path, callback_deps, base_config):
        model = _TinyModel()

        checkpoint_data = {
            "state_dict": {
                "linear.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "buf": torch.tensor([5.0]),
            }
        }

        monkeypatch.setattr("torch.load", lambda path: checkpoint_data)
        monkeypatch.setattr(_MODULE_PATH + ".select_with_path", lambda *, obj, path: obj)

        cb = EmaCallback(callback_config=SimpleNamespace(**base_config), model=model, **callback_deps)

        resumption_paths = Mock(checkpoint_path=tmp_path)
        cb.resume_from_checkpoint(resumption_paths, model)

        # Verify loaded data:
        assert torch.equal(cb.parameters[(None, 0.9)]["linear.weight"], checkpoint_data["state_dict"]["linear.weight"])
        assert torch.equal(cb.buffers[None]["buf"], checkpoint_data["state_dict"]["buf"])
        assert cb._was_resumed is True

    def test_skips_operations_on_non_rank0(self, monkeypatch, callback_deps, base_config):
        monkeypatch.setattr(_MODULE_PATH + ".is_rank0", lambda: False)

        model = _TinyModel()
        base_config["save_weights"] = True

        cb = EmaCallback(callback_config=SimpleNamespace(**base_config), model=model, **callback_deps)

        cb.before_training()
        assert len(cb.parameters) == 0

        cb.track_after_update_step()
        cb.periodic_callback(interval_type="update", update_counter=SimpleNamespace(cur_iteration="u=10"))

        callback_deps["checkpoint_writer"].save_model_checkpoint.assert_not_called()

    def test_apply_ema_modifies_in_place(self, monkeypatch, callback_deps, base_config):
        monkeypatch.setattr(_MODULE_PATH + ".is_rank0", lambda: True)
        monkeypatch.setattr(_MODULE_PATH + ".select_with_path", lambda *, obj, path: obj)

        model = _TinyModel()
        base_config["target_factors"] = [0.75]

        with torch.no_grad():
            model.linear.weight.fill_(4.0)

        cb = EmaCallback(callback_config=SimpleNamespace(**base_config), model=model, **callback_deps)
        cb.before_training()

        shadow = cb.parameters[(None, 0.75)]["linear.weight"]
        data_ptr_before = shadow.data_ptr()

        with torch.no_grad():
            model.linear.weight.fill_(8.0)

        cb.apply_ema(model, None, 0.75)

        # Verify same memory location (in-place):
        assert shadow.data_ptr() == data_ptr_before
        assert torch.allclose(shadow, torch.full_like(shadow, 5.0))  # 0.75 * 4 + 0.25 * 8
