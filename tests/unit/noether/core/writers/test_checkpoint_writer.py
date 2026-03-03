#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.nn.parallel import DistributedDataParallel

from noether.core.models import CompositeModel, Model
from noether.core.providers import PathProvider
from noether.core.types import CheckpointKeys
from noether.core.utils.training import TrainingIteration, UpdateCounter
from noether.core.writers.checkpoint_writer import CheckpointWriter

_MODULE_PATH = "noether.core.writers.checkpoint_writer"


def _make_writer(tmp_path: Path, run_id: str = "run-001") -> CheckpointWriter:
    path_provider = PathProvider(output_root_path=tmp_path, run_id=run_id)
    start = TrainingIteration(epoch=0, update=0, sample=0)
    end = TrainingIteration(epoch=5, update=None, sample=None)
    uc = UpdateCounter(start_iteration=start, end_iteration=end, updates_per_epoch=10, effective_batch_size=4)
    uc.cur_iteration = TrainingIteration(epoch=2, update=20, sample=80)

    return CheckpointWriter(path_provider=path_provider, update_counter=uc)


def _make_model(name: str = "encoder", is_frozen: bool = False, has_optimizer: bool = True) -> MagicMock:
    model = MagicMock(spec=Model)
    model.name = name
    model.is_frozen = is_frozen
    model.state_dict.return_value = {"weight": torch.tensor([1.0, 2.0])}
    model.model_config = MagicMock()
    model.model_config.model_dump.return_value = {"kind": "encoder", "dim": 64}
    model.model_config.config_kind = "my_project.EncoderConfig"
    if has_optimizer:
        model.optimizer = MagicMock()
        model.optimizer.state_dict.return_value = {"lr": 0.001}
    else:
        model.optimizer = None
    return model


class TestSaveModelCheckpoint:
    def test_saves_file_with_correct_name(self, tmp_path):
        writer = _make_writer(tmp_path)
        writer.save_model_checkpoint(
            model_name="encoder",
            checkpoint_tag="E2_U20_S80",
            state_dict={"w": torch.tensor(1.0)},
        )
        expected = writer.path_provider.checkpoint_path / "encoder_cp=E2_U20_S80_model.th"
        assert expected.exists()

    def test_checkpoint_contains_required_keys(self, tmp_path):
        writer = _make_writer(tmp_path)
        writer.save_model_checkpoint(
            model_name="encoder",
            checkpoint_tag="E2_U20_S80",
            state_dict={"w": torch.tensor(1.0)},
        )
        path = writer.path_provider.checkpoint_path / "encoder_cp=E2_U20_S80_model.th"
        checkpoint_data = torch.load(path, weights_only=False)
        assert CheckpointKeys.STATE_DICT in checkpoint_data
        assert CheckpointKeys.CHECKPOINT_TAG in checkpoint_data
        assert CheckpointKeys.TRAINING_ITERATION in checkpoint_data
        assert CheckpointKeys.RUN_ID in checkpoint_data

    def test_state_dict_roundtrip(self, tmp_path):
        writer = _make_writer(tmp_path)
        sd = {"layer.weight": torch.tensor([1.0, 2.0, 3.0])}
        writer.save_model_checkpoint(model_name="dec", checkpoint_tag="latest", state_dict=sd)
        path = writer.path_provider.checkpoint_path / "dec_cp=latest_model.th"
        checkpoint_data = torch.load(path, weights_only=False)
        assert torch.equal(checkpoint_data[CheckpointKeys.STATE_DICT]["layer.weight"], sd["layer.weight"])

    def test_checkpoint_tag_stored_as_string(self, tmp_path):
        writer = _make_writer(tmp_path)
        writer.save_model_checkpoint(model_name="m", checkpoint_tag="latest", state_dict={})
        path = writer.path_provider.checkpoint_path / "m_cp=latest_model.th"
        checkpoint_data = torch.load(path, weights_only=False)
        assert checkpoint_data[CheckpointKeys.CHECKPOINT_TAG] == "latest"

    def test_training_iteration_matches_counter(self, tmp_path):
        writer = _make_writer(tmp_path)
        writer.save_model_checkpoint(model_name="m", checkpoint_tag="tag", state_dict={})
        path = writer.path_provider.checkpoint_path / "m_cp=tag_model.th"
        checkpoint_data = torch.load(path, weights_only=False)
        ti = checkpoint_data[CheckpointKeys.TRAINING_ITERATION]
        assert ti["epoch"] == 2
        assert ti["update"] == 20
        assert ti["sample"] == 80

    def test_run_id_matches_provider(self, tmp_path):
        writer = _make_writer(tmp_path, run_id="abc-123")
        writer.save_model_checkpoint(model_name="m", checkpoint_tag="t", state_dict={})
        path = writer.path_provider.checkpoint_path / "m_cp=t_model.th"
        checkpoint_data = torch.load(path, weights_only=False)
        assert checkpoint_data[CheckpointKeys.RUN_ID] == "abc-123"

    def test_model_config_included_when_provided(self, tmp_path):
        writer = _make_writer(tmp_path)
        config = MagicMock()
        config.model_dump.return_value = {"dim": 128}
        config.config_kind = "my.Config"
        writer.save_model_checkpoint(model_name="m", checkpoint_tag="t", state_dict={}, model_config=config)
        path = writer.path_provider.checkpoint_path / "m_cp=t_model.th"
        checkpoint_data = torch.load(path, weights_only=False)
        assert checkpoint_data[CheckpointKeys.MODEL_CONFIG] == {"dim": 128}
        assert checkpoint_data[CheckpointKeys.CONFIG_KIND] == "my.Config"

    def test_model_config_absent_when_none(self, tmp_path):
        writer = _make_writer(tmp_path)
        writer.save_model_checkpoint(model_name="m", checkpoint_tag="t", state_dict={}, model_config=None)
        path = writer.path_provider.checkpoint_path / "m_cp=t_model.th"
        checkpoint_data = torch.load(path, weights_only=False)
        assert CheckpointKeys.MODEL_CONFIG not in checkpoint_data
        assert CheckpointKeys.CONFIG_KIND not in checkpoint_data

    def test_model_config_dump_error_raises_runtime(self, tmp_path):
        writer = _make_writer(tmp_path)
        config = MagicMock()
        config.model_dump.side_effect = TypeError("bad")
        with pytest.raises(RuntimeError, match="unexpected error"):
            writer.save_model_checkpoint(model_name="m", checkpoint_tag="t", state_dict={}, model_config=config)

    def test_model_info_included_in_filename(self, tmp_path):
        writer = _make_writer(tmp_path)
        writer.save_model_checkpoint(model_name="enc", checkpoint_tag="E1", state_dict={}, model_info="ema")
        expected = writer.path_provider.checkpoint_path / "enc_ema_cp=E1_model.th"
        assert expected.exists()

    def test_model_info_none_excluded_from_filename(self, tmp_path):
        writer = _make_writer(tmp_path)
        writer.save_model_checkpoint(model_name="enc", checkpoint_tag="E1", state_dict={}, model_info=None)
        expected = writer.path_provider.checkpoint_path / "enc_cp=E1_model.th"
        assert expected.exists()

    def test_extra_kwargs_included_in_checkpoint(self, tmp_path):
        writer = _make_writer(tmp_path)
        writer.save_model_checkpoint(model_name="m", checkpoint_tag="t", state_dict={}, custom_key="hello")
        path = writer.path_provider.checkpoint_path / "m_cp=t_model.th"
        checkpoint_data = torch.load(path, weights_only=False)
        assert checkpoint_data["custom_key"] == "hello"


class TestSaveSeparateModelsModel:
    def test_saves_weights_and_optim(self, tmp_path):
        writer = _make_writer(tmp_path)
        model = _make_model("enc")
        writer._save_separate_models(
            model=model,
            checkpoint_tag="E2",
            save_weights=True,
            save_optim=True,
            save_latest_weights=False,
            save_latest_optim=False,
            model_names_to_save=None,
            save_frozen_weights=True,
        )
        checkpoint_path = writer.path_provider.checkpoint_path
        assert (checkpoint_path / "enc_cp=E2_model.th").exists()
        assert (checkpoint_path / "enc_cp=E2_optim.th").exists()

    def test_save_latest_creates_latest_files(self, tmp_path):
        writer = _make_writer(tmp_path)
        model = _make_model("enc")
        writer._save_separate_models(
            model=model,
            checkpoint_tag="E2",
            save_weights=False,
            save_optim=False,
            save_latest_weights=True,
            save_latest_optim=True,
            model_names_to_save=None,
            save_frozen_weights=True,
        )
        checkpoint_path = writer.path_provider.checkpoint_path
        assert (checkpoint_path / "enc_cp=latest_model.th").exists()
        assert (checkpoint_path / "enc_cp=latest_optim.th").exists()
        assert not (checkpoint_path / "enc_cp=E2_model.th").exists()

    def test_frozen_model_skipped_when_flag_false(self, tmp_path):
        writer = _make_writer(tmp_path)
        model = _make_model("enc", is_frozen=True)
        writer._save_separate_models(
            model=model,
            checkpoint_tag="E2",
            save_weights=True,
            save_optim=True,
            save_latest_weights=False,
            save_latest_optim=False,
            model_names_to_save=None,
            save_frozen_weights=False,
        )
        checkpoint_path = writer.path_provider.checkpoint_path
        assert not (checkpoint_path / "enc_cp=E2_model.th").exists()

    def test_frozen_model_saved_when_flag_true(self, tmp_path):
        writer = _make_writer(tmp_path)
        model = _make_model("enc", is_frozen=True)
        writer._save_separate_models(
            model=model,
            checkpoint_tag="E2",
            save_weights=True,
            save_optim=False,
            save_latest_weights=False,
            save_latest_optim=False,
            model_names_to_save=None,
            save_frozen_weights=True,
        )
        checkpoint_path = writer.path_provider.checkpoint_path
        assert (checkpoint_path / "enc_cp=E2_model.th").exists()

    def test_model_names_filter(self, tmp_path):
        writer = _make_writer(tmp_path)
        model = _make_model("enc")
        writer._save_separate_models(
            model=model,
            checkpoint_tag="E2",
            save_weights=True,
            save_optim=False,
            save_latest_weights=False,
            save_latest_optim=False,
            model_names_to_save=["dec"],  # enc is not in the list
            save_frozen_weights=True,
        )
        checkpoint_path = writer.path_provider.checkpoint_path
        assert not (checkpoint_path / "enc_cp=E2_model.th").exists()

    def test_no_optimizer_skips_optim_save(self, tmp_path):
        writer = _make_writer(tmp_path)
        model = _make_model("enc", has_optimizer=False)
        writer._save_separate_models(
            model=model,
            checkpoint_tag="E2",
            save_weights=True,
            save_optim=True,
            save_latest_weights=False,
            save_latest_optim=False,
            model_names_to_save=None,
            save_frozen_weights=True,
        )
        checkpoint_path = writer.path_provider.checkpoint_path
        assert (checkpoint_path / "enc_cp=E2_model.th").exists()
        assert not (checkpoint_path / "enc_cp=E2_optim.th").exists()

    def test_model_name_override(self, tmp_path):
        writer = _make_writer(tmp_path)
        model = _make_model("enc")
        writer._save_separate_models(
            model=model,
            checkpoint_tag="E2",
            save_weights=True,
            save_optim=False,
            save_latest_weights=False,
            save_latest_optim=False,
            model_names_to_save=None,
            save_frozen_weights=True,
            model_name="autoenc.enc",
        )
        checkpoint_path = writer.path_provider.checkpoint_path
        assert (checkpoint_path / "autoenc.enc_cp=E2_model.th").exists()

    def test_none_model_is_noop(self, tmp_path):
        writer = _make_writer(tmp_path)
        # Should not raise:
        writer._save_separate_models(
            model=None,
            checkpoint_tag="E2",
            save_weights=True,
            save_optim=True,
            save_latest_weights=False,
            save_latest_optim=False,
            model_names_to_save=None,
            save_frozen_weights=True,
        )

    def test_ddp_model_raises(self, tmp_path):
        writer = _make_writer(tmp_path)
        model = MagicMock(spec=DistributedDataParallel)
        with pytest.raises(RuntimeError, match="DistributedDataParallel"):
            writer._save_separate_models(
                model=model,
                checkpoint_tag="E2",
                save_weights=True,
                save_optim=True,
                save_latest_weights=False,
                save_latest_optim=False,
                model_names_to_save=None,
                save_frozen_weights=True,
            )


class TestSaveSeparateModelsComposite:
    def test_composite_saves_all_submodels(self, tmp_path):
        writer = _make_writer(tmp_path)
        sub_enc = _make_model("enc")
        sub_dec = _make_model("dec")

        composite = MagicMock(spec=CompositeModel)
        composite.name = "autoenc"
        composite.submodels = {"encoder": sub_enc, "decoder": sub_dec}

        writer._save_separate_models(
            model=composite,
            checkpoint_tag="E2",
            save_weights=True,
            save_optim=False,
            save_latest_weights=False,
            save_latest_optim=False,
            model_names_to_save=None,
            save_frozen_weights=True,
        )
        checkpoint_path = writer.path_provider.checkpoint_path
        assert (checkpoint_path / "autoenc.encoder_cp=E2_model.th").exists()
        assert (checkpoint_path / "autoenc.decoder_cp=E2_model.th").exists()

    def test_composite_with_none_submodel(self, tmp_path):
        writer = _make_writer(tmp_path)
        sub_enc = _make_model("enc")

        composite = MagicMock(spec=CompositeModel)
        composite.name = "autoenc"
        composite.submodels = {"encoder": sub_enc, "decoder": None}

        # Should not raise:
        writer._save_separate_models(
            model=composite,
            checkpoint_tag="E2",
            save_weights=True,
            save_optim=False,
            save_latest_weights=False,
            save_latest_optim=False,
            model_names_to_save=None,
            save_frozen_weights=True,
        )
        checkpoint_path = writer.path_provider.checkpoint_path
        assert (checkpoint_path / "autoenc.encoder_cp=E2_model.th").exists()

    def test_unknown_model_type_raises(self, tmp_path):
        writer = _make_writer(tmp_path)
        model = MagicMock()  # neither Model nor CompositeModel
        with pytest.raises(NotImplementedError):
            writer._save_separate_models(
                model=model,
                checkpoint_tag="E2",
                save_weights=True,
                save_optim=False,
                save_latest_weights=False,
                save_latest_optim=False,
                model_names_to_save=None,
                save_frozen_weights=True,
            )


class TestSave:
    def test_save_on_rank0_writes_files(self, tmp_path):
        writer = _make_writer(tmp_path)
        model = _make_model("enc")

        with patch(_MODULE_PATH + ".is_rank0", return_value=True):
            writer.save(model, checkpoint_tag="E2", save_weights=True, save_optim=True)

        checkpoint_path = writer.path_provider.checkpoint_path
        assert (checkpoint_path / "enc_cp=E2_model.th").exists()
        assert (checkpoint_path / "enc_cp=E2_optim.th").exists()

    def test_save_not_rank0_writes_nothing(self, tmp_path):
        writer = _make_writer(tmp_path)
        model = _make_model("enc")

        with patch(_MODULE_PATH + ".is_rank0", return_value=False):
            writer.save(model, checkpoint_tag="E2", save_weights=True, save_optim=True)

        checkpoint_path = writer.path_provider.checkpoint_path
        assert not (checkpoint_path / "enc_cp=E2_model.th").exists()

    def test_save_with_trainer_writes_trainer_state(self, tmp_path):
        writer = _make_writer(tmp_path)
        model = _make_model("enc")
        trainer = MagicMock()
        trainer.state_dict.return_value = {"cb": []}

        with patch(_MODULE_PATH + ".is_rank0", return_value=True):
            writer.save(model, checkpoint_tag="E2", trainer=trainer, save_weights=True, save_optim=True)

        checkpoint_path = writer.path_provider.checkpoint_path
        trainer_path = checkpoint_path / "enc_cp=E2_trainer.th"
        assert trainer_path.exists()
        state_dict = torch.load(trainer_path, weights_only=False)
        assert state_dict == {"cb": []}

    def test_save_latest_flags_create_latest_trainer(self, tmp_path):
        writer = _make_writer(tmp_path)
        model = _make_model("enc")
        trainer = MagicMock()
        trainer.state_dict.return_value = {"cb": []}

        with patch(_MODULE_PATH + ".is_rank0", return_value=True):
            writer.save(
                model,
                checkpoint_tag="E2",
                trainer=trainer,
                save_weights=False,
                save_optim=False,
                save_latest_weights=True,
                save_latest_optim=True,
            )

        checkpoint_path = writer.path_provider.checkpoint_path
        assert (checkpoint_path / "enc_cp=latest_trainer.th").exists()
        # No E2 trainer file since save_weights=False and save_optim=False:
        assert not (checkpoint_path / "enc_cp=E2_trainer.th").exists()

    def test_save_no_trainer_skips_trainer_file(self, tmp_path):
        writer = _make_writer(tmp_path)
        model = _make_model("enc")

        with patch(_MODULE_PATH + ".is_rank0", return_value=True):
            writer.save(model, checkpoint_tag="E2", trainer=None, save_weights=True, save_optim=False)

        checkpoint_path = writer.path_provider.checkpoint_path
        assert not (checkpoint_path / "enc_cp=E2_trainer.th").exists()

    def test_trainer_state_dict_called_on_all_ranks(self, tmp_path):
        """trainer.state_dict() must be called before the rank0 check (for gathering random states)."""
        writer = _make_writer(tmp_path)
        model = _make_model("enc")
        trainer = MagicMock()
        trainer.state_dict.return_value = {}

        with patch(_MODULE_PATH + ".is_rank0", return_value=False):
            writer.save(model, checkpoint_tag="E2", trainer=trainer, save_weights=True, save_optim=True)

        trainer.state_dict.assert_called_once()
