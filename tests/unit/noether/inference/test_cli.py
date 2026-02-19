#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig

# We need to patch setup_hydra BEFORE importing the module to avoid running it:
with patch("noether.training.cli.setup_hydra"):
    from noether.inference.cli.main_inference import main as inference_main

_MODULE_PATH = "noether.inference.cli.main_inference"


@patch(_MODULE_PATH + ".InferenceRunner")
@patch(_MODULE_PATH + ".OmegaConf")
@patch(_MODULE_PATH + ".yaml")
@patch("builtins.open")
@patch(_MODULE_PATH + ".Path")
@patch(_MODULE_PATH + ".sys")
@patch(_MODULE_PATH + ".os")
@patch(_MODULE_PATH + ".hydra")
class TestMainInference:
    def test_main_success_path(
        self,
        mock_hydra,
        mock_os,
        mock_sys,
        mock_path_cls,
        mock_open,
        mock_yaml,
        mock_omegaconf,
        mock_runner_cls,
    ):
        """
        Test the successful execution path:
        1. Loads training config.
        2. Merges with inference config.
        3. Sets resume parameters correctly.
        4. runs InferenceRunner.
        """
        mock_hydra.utils.get_original_cwd.return_value = "/original/cwd"

        mock_input_dir = MagicMock()
        mock_input_dir.name = "run_folder_name"
        mock_input_dir.resolve.return_value = mock_input_dir

        # Mock hp_resolved.yaml existence:
        mock_hp_resolved = MagicMock()
        mock_hp_resolved.exists.return_value = True
        # Chain the division operator for: input_dir / run_id / stage_name / "hp_resolved.yaml"
        mock_input_dir.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_hp_resolved

        mock_path_cls.return_value = mock_input_dir

        # Mock YAML loading (the training config):
        train_config_dict = {
            "run_id": "train_run_id_123",
            "stage_name": "train_stage_A",
            "some_param": "original",
        }
        mock_yaml.full_load.return_value = train_config_dict

        mock_merged_config = MagicMock()
        mock_omegaconf.merge.return_value = mock_merged_config
        final_dict_config = {"final": "config"}
        mock_omegaconf.to_container.return_value = final_dict_config

        # Input Inference Config (passed via CLI):
        inference_config = DictConfig(
            {
                "input_dir": "/path/to/run",
                "some_param": "overwritten",
                "run_id": "some_run_id",
                "stage_name": "train_stage_A",
            }
        )

        inference_main.__wrapped__(inference_config)

        mock_os.chdir.assert_called_with("/original/cwd")
        mock_sys.path.insert.assert_called_with(0, "/original/cwd")
        mock_path_cls.assert_called_with("/path/to/run")

        mock_yaml.full_load.assert_called()
        mock_omegaconf.merge.assert_called_with(train_config_dict, inference_config)

        assert mock_merged_config.resume_run_id == "train_run_id_123"
        assert mock_merged_config.resume_stage_name == "train_stage_A"
        assert mock_merged_config.resume_checkpoint == "latest"

        mock_runner_cls.return_value.run.assert_called_once_with(final_dict_config)

    def test_missing_input_dir(self, mock_hydra, mock_os, mock_sys, *args):
        empty_config = DictConfig({})

        # Mock get_original_cwd to avoid errors before the check:
        mock_hydra.utils.get_original_cwd.return_value = "/cwd"

        with pytest.raises(ValueError, match="input_dir must be specified"):
            inference_main.__wrapped__(empty_config)

    def test_missing_hp_resolved(self, mock_hydra, mock_os, mock_sys, mock_path_cls, *args):
        mock_hydra.utils.get_original_cwd.return_value = "/cwd"

        mock_input_dir = MagicMock()
        mock_hp_resolved = MagicMock()
        mock_hp_resolved.exists.return_value = False  # missing file
        # Chain the division operator for: input_dir / run_id / stage_name / "hp_resolved.yaml"
        mock_input_dir.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_hp_resolved
        mock_path_cls.return_value.resolve.return_value = mock_input_dir

        config = DictConfig({"input_dir": "/bad/path", "run_id": "some_run_id"})

        with pytest.raises(FileNotFoundError, match="hp_resolved.yaml not found"):
            inference_main.__wrapped__(config)

    def test_resume_id_fallback_to_folder_name(
        self,
        mock_hydra,
        mock_os,
        mock_sys,
        mock_path_cls,
        mock_open,
        mock_yaml,
        mock_omegaconf,
        mock_runner_cls,
    ):
        mock_hydra.utils.get_original_cwd.return_value = "/cwd"

        mock_input_dir = MagicMock()
        mock_input_dir.resolve.return_value = mock_input_dir
        mock_input_dir.__truediv__.return_value.exists.return_value = True
        mock_path_cls.return_value = mock_input_dir

        mock_yaml.full_load.return_value = {}

        mock_merged_config = MagicMock()
        mock_omegaconf.merge.return_value = mock_merged_config

        config = DictConfig({"input_dir": ".", "run_id": "some_run_id", "stage_name": "some_stage"})

        inference_main.__wrapped__(config)

        assert mock_merged_config.resume_run_id == "some_run_id"
