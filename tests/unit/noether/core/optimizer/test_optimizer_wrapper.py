#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from noether.core.optimizer.optimizer_wrapper import OptimizerWrapper
from noether.core.schemas.optimizers import OptimizerConfig
from noether.core.utils.training.counter import UpdateCounter
from noether.core.utils.training.training_iteration import TrainingIteration


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)
        return x


class TestOptimizerWrapper:
    @pytest.fixture
    def model(self):
        return SimpleModel()

    @pytest.fixture
    def update_counter(self):
        return UpdateCounter(
            updates_per_epoch=10,
            effective_batch_size=1,
            start_iteration=TrainingIteration(epoch=0, update=0, sample=0),
            end_iteration=TrainingIteration(epoch=10),
        )

    def _create_mock_config(self):
        config = MagicMock(spec=OptimizerConfig)
        # Set all attributes accessed by OptimizerWrapper.__init__
        config.exclude_bias_from_weight_decay = True
        config.exclude_normalization_params_from_weight_decay = True
        config.param_group_modifiers_config = []
        config.schedule_config = None
        config.weight_decay_schedule = None
        config.clip_grad_norm = None
        config.clip_grad_value = None
        config.model_dump.return_value = {}
        return config

    @pytest.fixture
    def basic_config(self):
        return self._create_mock_config()

    def test_initialization(self, model, basic_config, update_counter):
        optimizer = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda param_groups: torch.optim.SGD(param_groups, lr=0.01, weight_decay=0.001),
            optim_wrapper_config=basic_config,
            update_counter=update_counter,
        )

        assert optimizer.torch_optim is not None
        assert len(optimizer.torch_optim.param_groups) > 0

    def test_exclude_bias_and_norm_from_weight_decay(self, model, basic_config, update_counter):
        optimizer = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda param_groups: torch.optim.SGD(param_groups, lr=0.01, weight_decay=0.001),
            optim_wrapper_config=basic_config,
            update_counter=update_counter,
        )

        for group in optimizer.torch_optim.param_groups:
            if "bn" in group["name"] or "bias" in group["name"]:
                assert group["weight_decay"] == 0.0

    def test_param_group_merging(self, model, basic_config, update_counter):
        param_groups = [
            {
                "params": [p for n, p in model.named_parameters() if n == "conv.weight"],
                "name": "conv.weight",
                "lr_scale": 1.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if n == "fc.weight"],
                "name": "fc.weight",
                "lr_scale": 1.0,
            },
        ]

        optimizer_wrapper = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda param_groups: torch.optim.SGD(param_groups, lr=0.01),
            optim_wrapper_config=basic_config,
            update_counter=update_counter,
        )

        merged_groups, merged_names = optimizer_wrapper._merge_groups_with_the_same_parameters(param_groups)

        assert len(merged_groups) > 0

    def test_has_param_with_grad(self, model, basic_config, update_counter):
        optimizer = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda param_groups: torch.optim.SGD(param_groups, lr=0.01),
            optim_wrapper_config=basic_config,
            update_counter=update_counter,
        )

        assert not optimizer._has_param_with_grad()

        for p in model.parameters():
            p.grad = torch.ones_like(p)

        assert optimizer._has_param_with_grad()

    def test_zero_grad(self, model, basic_config, update_counter):
        optimizer = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda param_groups: torch.optim.SGD(param_groups, lr=0.01),
            optim_wrapper_config=basic_config,
            update_counter=update_counter,
        )

        for p in model.parameters():
            p.grad = torch.ones_like(p)

        optimizer.zero_grad()

        for p in model.parameters():
            assert p.grad is None

    def test_state_dict(self, model, basic_config, update_counter):
        optimizer = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda param_groups: torch.optim.SGD(param_groups, lr=0.01),
            optim_wrapper_config=basic_config,
            update_counter=update_counter,
        )

        state_dict = optimizer.state_dict()
        assert "param_idx_to_name" in state_dict

    @patch("noether.core.optimizer.optimizer_wrapper.Bidict")
    def test_load_state_dict(self, mock_bidict, model, basic_config, update_counter):
        optimizer = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda param_groups: torch.optim.SGD(param_groups, lr=0.01),
            optim_wrapper_config=basic_config,
            update_counter=update_counter,
        )

        mock_state_dict = {
            "param_idx_to_name": {0: "conv.weight", 1: "conv.bias"},
            "state": {0: {"momentum_buffer": torch.ones(3, 16, 3, 3)}},
            "param_groups": [{"params": [0, 1], "lr": 0.01}],
        }

        mock_bidict_instance = MagicMock()
        mock_bidict.return_value = mock_bidict_instance

        with patch.object(optimizer.torch_optim, "load_state_dict") as mock_load:
            optimizer.load_state_dict(mock_state_dict)
            mock_load.assert_called_once()

    def test_step_with_gradient_clipping(self, model, update_counter):
        config = self._create_mock_config()
        config.clip_grad_norm = 1.0
        config.clip_grad_value = 0.5

        optimizer = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda p: torch.optim.SGD(p, lr=0.01),
            optim_wrapper_config=config,
            update_counter=update_counter,
        )

        for p in model.parameters():
            p.grad = torch.ones_like(p)

        # Mock the scaler to avoid internal state errors:
        mock_scaler = MagicMock(spec=torch.amp.GradScaler)

        with (
            patch("torch.nn.utils.clip_grad_norm_") as mock_norm,
            patch("torch.nn.utils.clip_grad_value_") as mock_value,
        ):
            optimizer.step(grad_scaler=mock_scaler)

            mock_scaler.unscale_.assert_called_once_with(optimizer.torch_optim)
            mock_norm.assert_called_once()
            mock_value.assert_called_once()
            mock_scaler.step.assert_called_once_with(optimizer.torch_optim)
            mock_scaler.update.assert_called_once()

    def test_apply_learning_rate_scaling(self, model, update_counter):
        config = self._create_mock_config()

        mock_modifier = MagicMock()
        mock_modifier.get_properties.return_value = {"lr_scale": 0.5}
        mock_modifier.was_applied_successfully.return_value = True

        with patch("noether.core.factory.Factory.create_list", return_value=[mock_modifier]):
            optimizer = OptimizerWrapper(
                model=model,
                torch_optim_ctor=lambda p: torch.optim.SGD(p, lr=0.1),
                optim_wrapper_config=config,
                update_counter=update_counter,
            )

            for group in optimizer.torch_optim.param_groups:
                assert group["lr"] == pytest.approx(0.05)
                assert group["original_lr"] == 0.1

    def test_schedule_step_lr_and_wd(self, model, update_counter):
        config = self._create_mock_config()
        # Explicitly disable exclusions so ALL params get the new WD
        config.exclude_bias_from_weight_decay = False
        config.exclude_normalization_params_from_weight_decay = False

        config.schedule_config = MagicMock()
        config.weight_decay_schedule = MagicMock()

        mock_lr_sched = MagicMock()
        mock_lr_sched.get_value.return_value = 0.002

        mock_wd_sched = MagicMock()
        mock_wd_sched.get_value.return_value = 0.05

        with patch("noether.core.factory.ScheduleFactory.create", side_effect=[mock_lr_sched, mock_wd_sched]):
            optimizer = OptimizerWrapper(
                model=model,
                torch_optim_ctor=lambda p: torch.optim.SGD(p, lr=0.01, weight_decay=0.01),
                optim_wrapper_config=config,
                update_counter=update_counter,
            )
            optimizer.schedule_step()

            for group in optimizer.torch_optim.param_groups:
                assert group["lr"] == 0.002
                assert group["weight_decay"] == 0.05

    def test_weight_decay_exclusion_logic_in_schedule(self, model, update_counter):
        config = self._create_mock_config()
        config.exclude_bias_from_weight_decay = True
        config.exclude_normalization_params_from_weight_decay = True
        config.weight_decay_schedule = MagicMock()

        mock_wd_sched = MagicMock()
        mock_wd_sched.get_value.return_value = 0.1

        with patch("noether.core.factory.ScheduleFactory.create", return_value=mock_wd_sched):
            optimizer = OptimizerWrapper(
                model=model,
                torch_optim_ctor=lambda p: torch.optim.SGD(p, lr=0.01, weight_decay=0.01),
                optim_wrapper_config=config,
                update_counter=update_counter,
            )
            optimizer.schedule_step()

            for group in optimizer.torch_optim.param_groups:
                # If bias, it should have been excluded from WD in init, and thus ignored by schedule:
                if "bias" in group.get("name", ""):
                    assert group["weight_decay"] == 0.0
                # Normal weights should get the scheduled value:
                elif "conv.weight" in group.get("name", ""):
                    assert group["weight_decay"] == 0.1

    def test_string_representations(self, model, update_counter):
        config = self._create_mock_config()
        config.clip_grad_norm = 1.0
        config.model_dump.return_value = {"clip_grad_norm": 1.0}

        optimizer = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda p: torch.optim.SGD(p, lr=0.01),
            optim_wrapper_config=config,
            update_counter=update_counter,
        )

        assert "SGD" in str(optimizer)
        assert "clip_grad_norm=1.0" in str(optimizer)
        assert "OptimizerWrapper" in repr(optimizer)

    def test_step_no_grads_skip(self, model, update_counter):
        config = self._create_mock_config()

        optimizer = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda p: torch.optim.SGD(p, lr=0.01),
            optim_wrapper_config=config,
            update_counter=update_counter,
        )

        scaler = MagicMock(spec=torch.amp.GradScaler)

        optimizer.zero_grad()
        optimizer.step(grad_scaler=scaler)
        # Should skip step:
        scaler.step.assert_not_called()

    def test_add_names_unsupported_type(self, model, update_counter):
        config = self._create_mock_config()

        optimizer = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda p: torch.optim.SGD(p, lr=0.01),
            optim_wrapper_config=config,
            update_counter=update_counter,
        )

        bad_group = [{"params": [], "custom_prop": "string_not_allowed"}]
        with pytest.raises(NotImplementedError):
            optimizer._add_names_to_param_groups(bad_group)
