#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import unittest

import torch

from noether.core.models import CompositeModel, Model, ModelBase
from noether.core.schemas.models.base import ModelBaseConfig
from noether.core.schemas.optimizers import OptimizerConfig


class TestCompositeModel(unittest.TestCase):
    class SimpleModel(Model):
        def __init__(self, model_config: ModelBaseConfig, is_frozen: bool = False, **kwargs):
            super().__init__(model_config=model_config, is_frozen=is_frozen, **kwargs)
            self.ff = torch.nn.Linear(10, 10)

        def forward(self, x):
            return self.ff(x)

    class SimpleCompositeModel(CompositeModel):
        def __init__(self, model1: Model, model2: Model, **kwargs):
            super().__init__(model_config=ModelBaseConfig(kind="composite", name="composite_model"), **kwargs)
            self.model1 = model1
            self.model2 = model2

        @property
        def submodels(self) -> dict[str, ModelBase]:
            return dict(model1=self.model1, model2=self.model2)

        def forward(self, x):
            x = self.model1(x)
            x = self.model2(x)
            return x

    def simple_composite_model(self, frozen_1: bool = False, frozen_2: bool = False) -> CompositeModel:
        return self.SimpleCompositeModel(
            model1=self.SimpleModel(
                model_config=ModelBaseConfig(
                    kind="single",
                    name="model1",
                    optimizer_config=OptimizerConfig(kind="torch.optim.AdamW", lr=1e-4) if not frozen_1 else None,
                ),
                is_frozen=frozen_1,
            ),
            model2=self.SimpleModel(
                model_config=ModelBaseConfig(
                    kind="single",
                    name="model2",
                    optimizer_config=OptimizerConfig(kind="torch.optim.AdamW", lr=1e-3) if not frozen_2 else None,
                ),
                is_frozen=frozen_2,
            ),
        )

    def test_is_frozen_if_submodels_are_frozen(self):
        model = self.simple_composite_model(frozen_1=True, frozen_2=True)

        self.assertTrue(model.model1.is_frozen)
        self.assertTrue(model.model2.is_frozen)
        self.assertTrue(model.is_frozen)
        with self.assertRaises(ValueError):
            self.SimpleCompositeModel(
                model1=self.SimpleModel(
                    model_config=ModelBaseConfig(
                        kind="single",
                        name="model1",
                        optimizer_config=OptimizerConfig(kind="torch.optim.AdamW", lr=1e-4),
                    ),
                    is_frozen=True,
                ),
                model2=self.SimpleModel(
                    model_config=ModelBaseConfig(
                        kind="single",
                        name="model2",
                        optimizer_config=OptimizerConfig(kind="torch.optim.AdamW", lr=1e-3),
                    ),
                    is_frozen=True,
                ),
            )

        model = self.simple_composite_model(frozen_1=True, frozen_2=False)
        self.assertTrue(model.model1.is_frozen)
        self.assertFalse(model.model2.is_frozen)
        self.assertFalse(model.is_frozen)

    def test_submodels_property(self):
        model = self.simple_composite_model()
        submodels = model.submodels
        self.assertIn("model1", submodels)
        self.assertIn("model2", submodels)
        self.assertIs(submodels["model1"], model.model1)
        self.assertIs(submodels["model2"], model.model2)

    def test_get_different_learning_rates(self):
        model = self.simple_composite_model()
        model.initialize_optimizer()
        named_models = model.submodels
        self.assertIn("model1", named_models)
        self.assertIn("model2", named_models)
        optim1 = named_models["model1"]._optim.torch_optim
        optim2 = named_models["model2"]._optim.torch_optim
        lr1 = optim1.param_groups[0]["lr"]
        lr2 = optim2.param_groups[0]["lr"]
        self.assertNotEqual(lr1, lr2)
        self.assertEqual(lr1, 1e-4)
        self.assertEqual(lr2, 1e-3)

    def test_get_named_models(self):
        model = self.simple_composite_model()
        named_models = model.get_named_models()
        self.assertIn("model1.model1", named_models)
        self.assertIn("model2.model2", named_models)
        self.assertIs(named_models["model1.model1"], model.model1)
        self.assertIs(named_models["model2.model2"], model.model2)

    def test_simple_forward(self):
        model = self.simple_composite_model()
        x = torch.randn(5, 10)
        output = model(x)
        self.assertEqual(output.shape, (5, 10))
        assert not torch.equal(output, x)

        model.initialize_optimizer()
        for name, param in model.model1.named_parameters():
            assert param.grad is None, f"Parameter {name} in model1 has no gradient!"

        for name, param in model.model2.named_parameters():
            assert param.grad is None, f"Parameter {name} in model2 has no gradient!"

        loss = output.sum()
        loss.backward()

        for name, param in model.model1.named_parameters():
            assert param.grad is not None, f"Parameter {name} in model1 has no gradient!"

        for name, param in model.model2.named_parameters():
            assert param.grad is not None, f"Parameter {name} in model2 has no gradient!"

        weights_before_step_model1 = {name: param.clone() for name, param in model.model1.named_parameters()}
        weights_before_step_model2 = {name: param.clone() for name, param in model.model2.named_parameters()}
        model.optimizer_step(grad_scaler=None)

        for name, param in model.model1.named_parameters():
            self.assertFalse(
                torch.equal(param, weights_before_step_model1[name]),
                f"Parameter {name} in model1 did not change after optimizer step!",
            )

        for name, param in model.model2.named_parameters():
            self.assertFalse(
                torch.equal(param, weights_before_step_model2[name]),
                f"Parameter {name} in model2 did not change after optimizer step!",
            )

        # test with frozen model
        frozen_model = self.simple_composite_model(frozen_1=True, frozen_2=True)
        frozen_model.initialize_optimizer()
        x = torch.randn(5, 10)
        output = frozen_model(x)
        self.assertEqual(output.shape, (5, 10))
        assert not torch.equal(output, x)

        weights_before_step_model1 = {name: param.clone() for name, param in frozen_model.model1.named_parameters()}
        weights_before_step_model2 = {name: param.clone() for name, param in frozen_model.model2.named_parameters()}

        loss = output.sum()
        loss.backward()

        frozen_model.optimizer_step(grad_scaler=None)
        for name, param in frozen_model.model1.named_parameters():
            self.assertTrue(
                torch.equal(param, weights_before_step_model1[name]),
                f"Parameter {name} in frozen model1 changed after optimizer step!",
            )

        for name, param in frozen_model.model2.named_parameters():
            self.assertTrue(
                torch.equal(param, weights_before_step_model2[name]),
                f"Parameter {name} in frozen model2 changed after optimizer step!",
            )

        partly_frozen_model = self.simple_composite_model(frozen_1=True, frozen_2=False)
        partly_frozen_model.initialize_optimizer()
        x = torch.randn(5, 10)
        output = partly_frozen_model(x)
        self.assertEqual(output.shape, (5, 10))
        assert not torch.equal(output, x)

        weights_before_step_model1 = {
            name: param.clone() for name, param in partly_frozen_model.model1.named_parameters()
        }
        weights_before_step_model2 = {
            name: param.clone() for name, param in partly_frozen_model.model2.named_parameters()
        }

        loss = output.sum()
        loss.backward()
        partly_frozen_model.optimizer_step(grad_scaler=None)
        for name, param in partly_frozen_model.model1.named_parameters():
            self.assertTrue(
                torch.equal(param, weights_before_step_model1[name]),
                f"Parameter {name} in model1 did not change after optimizer step!",
            )

        for name, param in partly_frozen_model.model2.named_parameters():
            self.assertFalse(
                torch.equal(param, weights_before_step_model2[name]),
                f"Parameter {name} in partly frozen model2 did not change after optimizer step!",
            )
