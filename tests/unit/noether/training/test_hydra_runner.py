#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from unittest.mock import ANY, MagicMock, patch

import pytest

from noether.core.schemas.schema import ConfigSchema
from noether.core.trackers.base import BaseTracker
from noether.training.runners.hydra_runner import HydraRunner

_MODULE_PATH = "noether.training.runners.hydra_runner"


class _DummyDataset:
    def __init__(self) -> None:
        self.pipeline = None


class TestHydraRunnerNaming:
    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            (["model.depth=4", "lr=0.001"], ["model.depth=4", "lr=0.001"]),
            (["+experiment=test"], ["experiment=test"]),
            (["accelerator=gpu", "devices=4", "tracker=wandb", "model.width=10"], ["model.width=10"]),
            (["dropout=0.5", "layers=2.0", "alpha=1."], ["dropout=0.5", "layers=2", "alpha=1"]),
            (["clip_grad_norm=1.0"], ["gclip=1"]),
        ],
    )
    def test_extract_name_from_overrides(self, overrides, expected):
        result = list(HydraRunner._extract_name_from_overrides(overrides))
        assert result == expected

    @patch(_MODULE_PATH + ".OmegaConf.to_container")
    @patch(_MODULE_PATH + ".hydra.core.hydra_config.HydraConfig")
    def test_derive_run_name_success(self, mock_hydra_config, mock_to_container):
        mock_to_container.return_value = ["model.name=resnet", "lr=0.01"]
        name = HydraRunner.derive_run_name("baseline")
        assert name == "baseline--model.name=resnet-lr=0.01"

    @patch(_MODULE_PATH + ".OmegaConf.to_container")
    @patch(_MODULE_PATH + ".hydra.core.hydra_config.HydraConfig")
    def test_derive_run_name_no_overrides(self, mock_hydra_config, mock_to_container):
        mock_to_container.return_value = []
        name = HydraRunner.derive_run_name("baseline")
        assert name == "baseline"


class TestHydraRunnerSetup:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock(spec=ConfigSchema)
        config.accelerator = "cpu"
        config.devices = 1
        config.debug = False
        config.run_id = None
        config.output_path = "/tmp/output"
        config.stage_name = "train"
        config.name = "test_run"
        config.seed = 123
        config.resume_run_id = None
        config.num_workers = 0
        config.store_code_in_output = False
        config.cudnn_benchmark = False
        config.cudnn_deterministic = False

        config.model = MagicMock()
        config.model.name = "test_model"

        config.datasets = {"train": MagicMock(pipeline=None)}
        config.trainer = MagicMock()
        config.tracker = MagicMock()

        config.model_dump.return_value = {}

        return config

    @patch(_MODULE_PATH + ".ModelBase", object)
    @patch(_MODULE_PATH + ".BaseTrainer", object)
    @patch(_MODULE_PATH + ".DatasetWrapper", object)
    @patch(_MODULE_PATH + ".Dataset", _DummyDataset)
    @patch(_MODULE_PATH + ".Hyperparameters")
    @patch(_MODULE_PATH + ".add_global_handlers")
    @patch(_MODULE_PATH + ".store_code_archive")
    @patch(_MODULE_PATH + ".set_seed")
    @patch(_MODULE_PATH + ".PathProvider")
    @patch(_MODULE_PATH + ".DatasetFactory")
    @patch(_MODULE_PATH + ".Factory")
    def test_setup_experiment_happy_path(
        self,
        mock_factory_cls,
        mock_dataset_factory_cls,
        mock_path_provider,
        mock_set_seed,
        mock_store_code,
        mock_add_handlers,
        mock_hyperparameters,
        mock_config,
    ):
        mock_factory = mock_factory_cls.return_value
        mock_dataset_factory = mock_dataset_factory_cls.return_value

        mock_dataset_factory.create.return_value = _DummyDataset()

        mock_tracker = MagicMock(spec=BaseTracker)
        mock_trainer = MagicMock()
        mock_model = MagicMock()

        mock_path_instance = mock_path_provider.return_value
        mock_path_instance.logfile_uri = "/tmp/test.log"
        mock_path_instance.run_output_path = MagicMock()

        mock_factory.create.side_effect = [mock_tracker, mock_trainer]
        mock_factory.instantiate.return_value = mock_model

        trainer, model, tracker, _ = HydraRunner.setup_experiment(device="cpu", config=mock_config)

        mock_set_seed.assert_called_with(123)
        mock_dataset_factory.create.assert_called()

        mock_factory.create.assert_any_call(
            mock_config.tracker,
            metric_property_provider=ANY,
            path_provider=ANY,
        )

        mock_factory.create.assert_any_call(
            mock_config.trainer,
            data_container=ANY,
            device="cpu",
            tracker=mock_tracker,
            path_provider=ANY,
            metric_property_provider=ANY,
        )

    @patch(_MODULE_PATH + ".ModelBase", object)
    @patch(_MODULE_PATH + ".BaseTrainer", object)
    @patch(_MODULE_PATH + ".DatasetWrapper", object)
    @patch(_MODULE_PATH + ".Dataset", _DummyDataset)
    @patch(_MODULE_PATH + ".Hyperparameters")
    @patch(_MODULE_PATH + ".add_global_handlers")
    @patch(_MODULE_PATH + ".PathProvider")
    @patch(_MODULE_PATH + ".DatasetFactory")
    @patch(_MODULE_PATH + ".Factory")
    def test_resume_logic(
        self,
        mock_factory_cls,
        mock_dataset_factory_cls,
        mock_path_provider_cls,
        mock_add_handlers,
        mock_hyperparameters,
        mock_config,
    ):
        mock_config.resume_run_id = "run_123"
        mock_config.resume_stage_name = "prev_stage"
        mock_config.resume_checkpoint = "latest"

        mock_path_instance = mock_path_provider_cls.return_value
        mock_path_instance.logfile_uri = "/tmp/test.log"

        mock_factory = mock_factory_cls.return_value
        mock_factory.create.side_effect = [MagicMock(spec=BaseTracker), MagicMock()]
        mock_factory.instantiate.return_value = MagicMock()

        mock_dataset_factory_cls.return_value.create.return_value = _DummyDataset()

        HydraRunner.setup_experiment(device="cpu", config=mock_config)

        mock_path_instance.with_run.assert_called_with(run_id="run_123", stage_name="prev_stage")
        mock_path_instance.link.assert_called()
        assert mock_config.trainer.initializer is not None
