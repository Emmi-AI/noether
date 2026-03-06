#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from noether.core.callbacks import CallbackBase
from noether.core.initializers.checkpoint import CheckpointInitializer
from noether.core.models import CompositeModel, Model, ModelBase
from noether.core.utils.training.training_iteration import TrainingIteration

if TYPE_CHECKING:  # import only for type checking to avoid circular imports
    from noether.training.trainers import BaseTrainer

from noether.core.schemas.initializers import ResumeInitializerConfig
from noether.core.types import CheckpointKeys
from noether.core.utils.model import compute_model_norm


class ResumeInitializer(CheckpointInitializer):
    """Initializes models/optimizers from a checkpoint ready for resuming training.
    Needs to be configured as part of the config by setting the resume_run_id in the root config (assumg the same output path is used).
    This initializer assumes that the previous run is going to be resumed for further training (i.e., trianing is not finished yet)

    For example (config snippet is part of the trainer config):

    .. code-block:: yaml

        resume_run_id: <previous_run_id>

    """

    def __init__(self, initializer_config: ResumeInitializerConfig, **kwargs):
        """
        Args:
            initializer_config: configuration for the initializer. See :class:`~noether.core.schemas.initializers.ResumeInitializerConfig` for available options.
            **kwargs: additional arguments to pass to the parent class.
        """
        super().__init__(initializer_config=initializer_config, **kwargs)

    def init_weights(self, model: ModelBase) -> None:
        """Initialize the model weights from the checkpoint.

        Args:
            model: the model to load the weights into.
        """
        self._init_weights(model.name, model)

    def _init_weights(self, name: str, model: ModelBase) -> None:
        """Initialize the model weights from the checkpoint.

        If the model is a CompositeModel, recursively initialize the weights of all Model submodels.

        Args:
            name: the name of the model.
            model: the model to load the weights into.
        """
        if isinstance(model, Model):
            model_name, checkpoint_uri = self._get_modelname_and_checkpoint_uri(
                model=model, model_name=name, file_type="model"
            )
            checkpoint = torch.load(checkpoint_uri, map_location=model.device, weights_only=True)
            if CheckpointKeys.STATE_DICT not in checkpoint:
                raise KeyError(f"state_dict not found in checkpoint {checkpoint_uri}")
            state_dict = checkpoint[CheckpointKeys.STATE_DICT]
            model_norm = compute_model_norm(model)
            model.load_state_dict(state_dict)
            if not model.is_frozen and compute_model_norm(model) == model_norm:
                raise RuntimeError(
                    f"Model weights of {model_name} have not changed after loading from checkpoint {checkpoint_uri}"
                )
            self.logger.info(f"loaded weights of {model_name} from {checkpoint_uri}")
        if isinstance(model, CompositeModel):
            for submodel_name, submodel in model.submodels.items():
                self._init_weights(name=f"{name}.{submodel_name}", model=submodel)

    def init_optimizer(self, model: ModelBase) -> None:
        """Initialize the optimizer for the model.

        Args:
            model: a model to initialize the optimizer for.
        """
        self._init_optimizer(name=model.name, model=model)

    def _init_optimizer(self, name: str, model: ModelBase) -> None:
        """Initialize the optimizer for the model.

        If the model is a CompositeModel, recursively initialize the optimizer of all Model submodels.

        Args:
            name: the name of the model.
            model: a model to initialize the optimizer for.
        """
        if isinstance(model, Model):
            if model.optimizer is None:
                # e.g. EMA target network doesn't have an optimizer
                self.logger.info(
                    f"skip loading optim from checkpoint '{self.checkpoint_tag}' for {name} ({model.name}) (optim is None)"
                )
            elif model.is_frozen:
                self.logger.info(
                    f"skip loading optim from checkpoint '{self.checkpoint_tag}' for {name}  ({model.name}) (is_frozen)"
                )
            else:
                model_name, checkpoint_uri = self._get_modelname_and_checkpoint_uri(
                    model=model, model_name=name, file_type="optim"
                )
                state_dict = torch.load(checkpoint_uri, map_location=model.device)
                model.optimizer.load_state_dict(state_dict)
                self.logger.info(f"loaded optimizer of {model_name} from {checkpoint_uri}")
        if isinstance(model, CompositeModel):
            for submodel_name, submodel in model.submodels.items():
                self._init_optimizer(name=f"{name}.{submodel_name}", model=submodel)

    def _get_trainer_ckpt_file(self) -> Path:
        """Get the checkpoint file for the trainer.

        Returns:
            ckpt_uri: the checkpoint file for the trainer.
        """
        return self._get_checkpoint_uri(prefix=f"{self.model_name}_cp=", suffix="_trainer.th")

    def start_checkpoint(self) -> TrainingIteration:
        """Get the start checkpoint for the model.

        Returns:
            TrainingIteration: the start checkpoint for the model.
        """
        if isinstance(self.checkpoint_tag, str):
            trainer_ckpt = torch.load(self._get_trainer_ckpt_file())
            self.logger.info(f"loaded checkpoint from trainer_state_dict: {trainer_ckpt}")
            return TrainingIteration.from_dict(trainer_ckpt[CheckpointKeys.TRAINING_ITERATION])
        else:
            return TrainingIteration.to_fully_specified_from_filenames(
                directory=self.path_provider.with_run(
                    run_id=self.run_id,
                    stage_name=self.stage_name,
                ).checkpoint_path.as_posix(),
                training_iteration=self.checkpoint_tag,
            )

    def init_trainer(self, trainer: BaseTrainer) -> None:
        """Initialize the trainer from the checkpoint.

        Args:
            trainer: the trainer to initialize.
        """
        checkpoint_uri = self._get_trainer_ckpt_file()
        trainer.load_state_dict(torch.load(checkpoint_uri))
        self.logger.info(f"loaded trainer checkpoint {checkpoint_uri}")

    def init_callbacks(self, callbacks: list[CallbackBase], model: ModelBase) -> None:
        """Initialize the callbacks from the checkpoint.

        Args:
            callbacks: the callbacks to initialize.
            model: the model to initialize the callbacks for.
        """
        trainer_state_dict: dict = torch.load(self._get_trainer_ckpt_file())
        callback_state_dicts = trainer_state_dict.pop(CheckpointKeys.CALLBACK_STATE_DICT)

        num_in_checkpoint = len(callback_state_dicts)
        num_current = len(callbacks)
        if num_in_checkpoint != num_current:
            # Callback count can differ between runs (e.g. EtaCallback is only added in TTY
            # environments, spawned multi-GPU processes lose TTY). Only stateful callbacks matter.
            num_stateful_ckpt = sum(1 for sd in callback_state_dicts if sd is not None)
            num_stateful_current = sum(1 for cb in callbacks if cb.state_dict() is not None)
            if num_stateful_ckpt != num_stateful_current:
                raise ValueError(
                    f"Number of stateful callbacks in checkpoint ({num_stateful_ckpt}) does not match "
                    f"number of stateful callbacks in current trainer ({num_stateful_current})"
                )
            self.logger.warning(
                f"Callback count mismatch: checkpoint has {num_in_checkpoint}, current trainer has {num_current}. "
                f"All {num_stateful_ckpt} stateful callbacks match - loading positionally."
            )

        for callback, state_dict in zip(callbacks, callback_state_dicts, strict=False):
            callback.load_state_dict(state_dict)
            callback.resume_from_checkpoint(
                resumption_paths=self.init_run_path_provider,
                model=model,
            )
