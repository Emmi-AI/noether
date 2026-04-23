#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import copy

import pytest
import torch
import torch.nn as nn

from noether.core.optimizer.muon_composite import MuonComposite


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)  # weight is 2D, bias is 1D
        self.bn = nn.BatchNorm1d(4)  # weight and bias are 1D

    def forward(self, x):
        return self.bn(self.linear(x))


def _build_param_groups(model):
    """Build one param group per named parameter, mimicking OptimizerWrapper."""
    groups = []
    for name, param in model.named_parameters():
        groups.append(
            {
                "params": [param],
                "name": name,
            }
        )
    return groups


class TestMuonComposite:
    @pytest.fixture
    def model(self):
        return SimpleModel()

    @pytest.fixture
    def param_groups(self, model):
        return _build_param_groups(model)

    def test_parameter_splitting(self, model, param_groups):
        """2D params go to Muon, 1D params go to secondary."""
        opt = MuonComposite(param_groups, lr=1e-3)

        muon_params = [p for g in opt._muon.param_groups for p in g["params"]]
        secondary_params = [p for g in opt._secondary.param_groups for p in g["params"]]

        expected_muon = {p for n, p in model.named_parameters() if p.ndim >= 2}
        expected_secondary = {p for n, p in model.named_parameters() if p.ndim < 2}

        assert set(muon_params) == expected_muon
        assert set(secondary_params) == expected_secondary
        # No param should appear in both
        assert set(muon_params).isdisjoint(set(secondary_params))

    def test_default_secondary_is_lion(self, param_groups):
        """When no secondary kind is specified, Lion is used."""
        opt = MuonComposite(param_groups, lr=1e-3)
        assert type(opt._secondary).__name__ == "Lion"

    def test_custom_secondary_optimizer(self, param_groups):
        """A custom secondary optimizer kind should be respected."""
        opt = MuonComposite(
            param_groups,
            lr=1e-3,
            secondary={"kind": "torch.optim.AdamW", "lr": 5e-4},
        )
        assert isinstance(opt._secondary, torch.optim.AdamW)

    def test_secondary_inherits_lr_and_wd(self, param_groups):
        """Secondary should inherit primary lr/wd when not explicitly set."""
        opt = MuonComposite(param_groups, lr=0.01, weight_decay=0.05)

        for g in opt._secondary.param_groups:
            assert g["lr"] == 0.01
            assert g["weight_decay"] == 0.05

    def test_secondary_explicit_lr_overrides(self, param_groups):
        """Explicit secondary lr/wd should override inherited values."""
        opt = MuonComposite(
            param_groups,
            lr=0.01,
            weight_decay=0.05,
            secondary={"lr": 0.001, "weight_decay": 0.1},
        )

        for g in opt._secondary.param_groups:
            assert g["lr"] == 0.001
            assert g["weight_decay"] == 0.1

    def test_step_updates_params(self, model, param_groups):
        """A step with gradients should actually change parameters."""
        opt = MuonComposite(param_groups, lr=1e-2)

        params_before = {n: p.clone() for n, p in model.named_parameters()}

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        changed = False
        for n, p in model.named_parameters():
            if not torch.equal(p.data, params_before[n]):
                changed = True
                break
        assert changed, "step() did not update any parameters"

    def test_step_with_closure(self, model, param_groups):
        """step(closure) should call the closure and return its value."""
        opt = MuonComposite(param_groups, lr=1e-2)

        def closure():
            x = torch.randn(2, 8)
            return model(x).sum()

        loss = opt.step(closure=closure)
        assert loss is not None

    def test_zero_grad_clears_all(self, model, param_groups):
        """zero_grad should clear gradients on all parameters."""
        opt = MuonComposite(param_groups, lr=1e-3)

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()

        # Verify grads exist
        assert any(p.grad is not None for p in model.parameters())

        opt.zero_grad()

        for p in model.parameters():
            assert p.grad is None

    def test_state_dict_indices_are_unique(self, model, param_groups):
        """state_dict must have globally unique param indices (no collisions)."""
        opt = MuonComposite(param_groups, lr=1e-3)

        # Do a step so state is populated
        x = torch.randn(2, 8)
        model(x).sum().backward()
        opt.step()

        sd = opt.state_dict()

        all_indices = []
        for g in sd["param_groups"]:
            all_indices.extend(g["params"])

        assert len(all_indices) == len(set(all_indices)), "Duplicate param indices in state_dict"

    def test_state_dict_covers_all_params(self, model, param_groups):
        """state_dict param_groups should reference every parameter."""
        opt = MuonComposite(param_groups, lr=1e-3)

        sd = opt.state_dict()
        total_params_in_sd = sum(len(g["params"]) for g in sd["param_groups"])
        total_params_in_model = sum(1 for _ in model.parameters())

        assert total_params_in_sd == total_params_in_model

    def test_load_state_dict_roundtrip(self, model, param_groups):
        """save -> load -> step should produce the same result as continuous stepping."""
        opt = MuonComposite(param_groups, lr=1e-2)
        torch.manual_seed(0)

        # Step once to populate optimizer state
        x = torch.randn(2, 8)
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()

        # Save state with deepcopy because state_dict() returns references to live
        # buffers that get mutated in-place by subsequent steps.
        sd = copy.deepcopy(opt.state_dict())
        params_after_first_step = {n: p.clone() for n, p in model.named_parameters()}

        # Step again
        torch.manual_seed(1)
        x2 = torch.randn(2, 8)
        model(x2).sum().backward()
        opt.step()
        params_after_second_step = {n: p.clone() for n, p in model.named_parameters()}

        # Restore model and optimizer to post-first-step state
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(params_after_first_step[n])

        opt2 = MuonComposite(_build_param_groups(model), lr=1e-2)
        opt2.load_state_dict(sd)

        # Replay the same second step
        opt2.zero_grad()
        torch.manual_seed(1)
        x3 = torch.randn(2, 8)
        model(x3).sum().backward()
        opt2.step()

        for n, p in model.named_parameters():
            torch.testing.assert_close(
                p.data,
                params_after_second_step[n],
                msg=f"Parameter {n} diverged after load_state_dict roundtrip",
            )

    def test_scheduler_mutation_propagates(self, param_groups):
        """External lr changes to param_groups should reach internal optimizers."""
        opt = MuonComposite(param_groups, lr=1e-3)

        new_lr = 5e-4
        for g in opt.param_groups:
            g["lr"] = new_lr

        # Verify the internal optimizers see the change
        for g in opt._muon.param_groups:
            assert g["lr"] == new_lr
        for g in opt._secondary.param_groups:
            assert g["lr"] == new_lr
