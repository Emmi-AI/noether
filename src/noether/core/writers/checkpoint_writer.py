#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch.nn.parallel import DistributedDataParallel

from noether.core.distributed import is_rank0
from noether.core.models import CompositeModel, Model
from noether.core.providers import PathProvider
from noether.core.schemas.models import ModelBaseConfig
from noether.core.types import CheckpointKeys
from noether.core.utils.training import UpdateCounter

if TYPE_CHECKING:
    from noether.core.models import ModelBase
    from noether.training.trainers import BaseTrainer


class CheckpointWriter:
    """Class to easily write checkpoints in a structured way to the disk.

    Each `Model` will be stored in a separate file where additionally weights and optimizer state are also
    separate files. This allows flexible storing of model states without producing files that are never needed after
    training. For example, to resume runs, one need the model weights and optimizer states. However, storing
    optimizer states for all checkpoints is expensive as optimizer states are commonly 2x as large as only the weights.

    To illustrate the flexibility, consider the use-case of training an autoencoder model where the goal is to train
    a good encoder that should then be used for another task. This model is implemented via a class `Autoencoder` that
    inherits from `CompositeModel` and contains two submodels, an encoder and decoder (both which inherit from
    `Model`). During training, we want to store the following files to the disk:
    - The encoder weights after every 10 epochs to evaluate performance at various training lengths.
    - The latest weights and optimizer states of encoder and decoder to allow resuming a run if it crashes.
    The CheckpointWriter provides functionality to store the following files:
    - `ab_upt_cp=E10_... model.th`: encoder weights after 10 epochs
    - `ab_upt_cp=E20_... model.th`: encoder weights after 20 epochs
    - `ab_upt_cp=E30_... model.th`: encoder weights after 30 epochs
    - `ab_upt_cp=last_model.th`: latest encoder weights
    - `ab_upt_cp=last_optim.th`: latest encoder optimizer state


    Each model checkpoint is populated with metadata. Each checkpoint will be a dictionary containing the keys:
    - "state_dict": Weights of the model.
    - "model_config": The model configuration used to instantiate the model. A serialized dict of the pydantic model config.
    - "checkpoint_tag": The name of the checkpoint. E.g., E10_U200_S800 for a progress-based checkpoint or "latest" for a
      string-based checkpoint.
    - "training_iteration": The detailed information about training iteration as a dict with keys 'epoch', 'update', and 'sample'.
      E.g., for the "latest" checkpoint you would not know from which epoch the checkpoint is, therefore
      the "training_iteration" field of that checkpoint contains "E13_U..._S...".
    - "run_id": The ID of the run from which it was created.
    """

    def __init__(self, path_provider: PathProvider, update_counter: UpdateCounter):
        self.logger = logging.getLogger(type(self).__name__)
        self.path_provider = path_provider
        self.update_counter = update_counter

    def save_model_checkpoint(
        self,
        model_name: str,
        checkpoint: str,
        state_dict: dict[str, Any],
        model_config: ModelBaseConfig | None = None,
        model_info: str | None = None,
        **extra,
    ) -> None:
        """Save a checkpoint to disk.

        The ouput name of the checkpoint will be constructed as `{model_name}{model_info}_cp={checkpoint}_model.th` (where model_info is an optional string and will be empty if not provided).
        For example, if model_name is "autoencoder.encoder" and checkpoint is "E10_U200_S800", the output name will be "autoencoder.encoder_cp=E10_U200_S800_model.th".
        However, if we store different model that the current one that is trained, for example, the EMA model, we can also provide additional info in the output name.
        For example, if model_name is "autoencoder.encoder" and model_info is "ema", and checkpoint is "E10_U200_S800", the output name will be "autoencoder.encoder_ema_cp=E10_U200_S800_model.th".

        Args:
            model_name: Name of the model.
            checkpoint: Checkpoint tag, for example "latest" or "E10_U200_S800".
            state_dict: Model state dict to save.
            model_config: Model configuration. Defaults to None.
            model_info: Additional info to include in the output name. Defaults to None.
            **extra:

        Raises:
            RuntimeError: in case of an unexpected error while parsing `model_config`.
        """
        output_dict = {
            CheckpointKeys.STATE_DICT: state_dict,
            # CheckpointKeys.CHECKPOINT_TAG: str(checkpoint_tag), # TODO: fix this
            CheckpointKeys.TRAINING_ITERATION: dict(self.update_counter.cur_iteration),
            CheckpointKeys.RUN_ID: self.path_provider.run_id,
            **extra,
        }

        if model_config is not None:
            try:
                output_dict[CheckpointKeys.MODEL_CONFIG] = model_config.model_dump()
            except Exception as e:
                raise RuntimeError(f"An unexpected error occurred during model_dump: {e}") from e
            output_dict[CheckpointKeys.CONFIG_KIND] = model_config.config_kind

        # Construct model URI with optional model_info; follows structure: {model_name}_{model_info}_cp={checkpoint}_model.th
        model_info = f"_{model_info}" if model_info else ""
        model_uri = self.path_provider.checkpoint_path / f"{model_name}{model_info}_cp={checkpoint}_model.th"

        torch.save(output_dict, model_uri)
        self.logger.info(f"Saved model to {model_uri}")

    def save(
        self,
        model: ModelBase,
        checkpoint: str,
        trainer: BaseTrainer | None = None,
        save_weights: bool = True,
        save_optim: bool = True,
        save_latest_weights: bool = False,
        save_latest_optim: bool = False,
        model_names_to_save: list[str] | None = None,
        save_frozen_weights: bool = True,
        model_info: str | None = None,
    ) -> None:
        """Saves a model to the disk.

        Args:
            model: Model to save.
            checkpoint_tag: Checkpoint tag, for example "latest" or "E10_U200_S800".
            trainer: If defined, also stores the state_dict of the trainer (and callbacks).
            save_weights: If true, stores model weights.
            save_optim: If true, stores optimizer states.
            save_latest_weights: If true, also stores the weights with the checkpoint identifier "latest". This
              file will be repeatedly overwritten throughout a training procedure to save storage.
            save_latest_optim: If true, also stores the optimizer states with the checkpoint identifier "latest". This
              file will be repeatedly overwritten throughout a training procedure to save storage.
            model_names_to_save: If defined, only store some of the submodels of a `CompositeModel`.
            save_frozen_weights: If true, also stores the weights of frozen models.
        """

        # NOTE: this has to be called from all ranks because random states are gathered to rank0
        trainer_sd = trainer.state_dict() if trainer is not None else None

        if is_rank0():
            self._save_separate_models(
                model=model,
                checkpoint=checkpoint,
                save_weights=save_weights,
                save_optim=save_optim,
                save_latest_weights=save_latest_weights,
                save_latest_optim=save_latest_optim,
                model_names_to_save=model_names_to_save,
                save_frozen_weights=save_frozen_weights,
                model_info=model_info,
            )

            if trainer_sd is not None:
                save_requests = [
                    (save_weights or save_optim, checkpoint),
                    (save_latest_weights or save_latest_optim, "latest"),
                ]

                for should_save, tag in save_requests:
                    if should_save:
                        trainer_out_path = self.path_provider.checkpoint_path / f"{model.name}_cp={tag}_trainer.th"
                        torch.save(trainer_sd, trainer_out_path)
                        self.logger.info(f"saved trainer state_dict to {trainer_out_path}")

    def _save_separate_models(
        self,
        model: ModelBase,
        checkpoint: str,
        save_weights: bool,
        save_optim: bool,
        save_latest_weights: bool,
        save_latest_optim: bool,
        model_names_to_save: list[str] | None,
        save_frozen_weights: bool,
        model_info: str | None = None,
        model_name: str | None = None,
    ):
        if isinstance(model, DistributedDataParallel):
            raise RuntimeError("DistributedDataParallel models should be unwrapped before saving.")
        # composite models can have submodels that are none -> skip them
        if model is None:
            return

        if isinstance(model, Model):
            if model.is_frozen and not save_frozen_weights:
                return
            if model_names_to_save and model.name not in model_names_to_save:
                return

            # --- Save Weights ---
            weight_requests = [
                (save_weights, checkpoint),
                (save_latest_weights, "latest"),
            ]

            for should_save, tag in weight_requests:
                if should_save:
                    self.save_model_checkpoint(
                        model_name=model.name if model_name is None else model_name,
                        checkpoint=tag,
                        model_info=model_info,
                        state_dict=model.state_dict(),
                        model_config=getattr(model, "model_config", None),
                    )

            # --- Save Optimizer ---
            if model.optimizer is not None:
                optim_requests = [
                    (save_optim, checkpoint),
                    (save_latest_optim, "latest"),
                ]

                for should_save, tag in optim_requests:
                    if should_save:
                        optimizer_uri = (
                            self.path_provider.checkpoint_path
                            / f"{model.name if model_name is None else model_name}_cp={tag}_optim.th"
                        )
                        torch.save(model.optimizer.state_dict(), optimizer_uri)
                        self.logger.info(f"Saved {model.name} optimizer to {optimizer_uri}")

        elif isinstance(model, CompositeModel):
            for k, v in model.submodels.items():
                self._save_separate_models(
                    model=v,
                    model_name=f"{model.name}.{k}",
                    checkpoint=checkpoint,
                    model_info=model_info,
                    save_weights=save_weights,
                    save_optim=save_optim,
                    save_latest_weights=save_latest_weights,
                    save_latest_optim=save_latest_optim,
                    model_names_to_save=model_names_to_save,
                    save_frozen_weights=save_frozen_weights,
                )
        else:
            raise NotImplementedError
