#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from noether.core.initializers.base import InitializerBase
from noether.core.models import Model, ModelBase
from noether.core.schemas.initializers import CheckpointInitializerConfig
from noether.core.types import CheckpointKeys
from noether.core.utils.training.training_iteration import TrainingIteration  # fixme?


class CheckpointInitializer(InitializerBase):
    checkpoint: str | TrainingIteration

    """
    Base class to initialize models from checkpoints of previous runs. Should not be used directly, but inherited by other initializers such as PreviousRunInitializer or ResumeInitializer.
    """

    def __init__(
        self,
        initializer_config: CheckpointInitializerConfig,
        **kwargs,
    ):
        """

        Args:
            initializer_config: configuration for the initializer. See :class:`~noether.core.schemas.initializers.CheckpointInitializerConfig` for available options.
            **kwargs: additional arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        self.run_id = initializer_config.run_id
        self.model_name = initializer_config.model_name
        self.load_optim = initializer_config.load_optim
        self.model_info = initializer_config.model_info
        self.pop_ckpt_kwargs_keys = initializer_config.pop_ckpt_kwargs_keys or []
        self.stage_name = initializer_config.stage_name
        self.init_run_path_provider = self.path_provider.with_run(
            run_id=self.run_id,
            stage_name=self.stage_name,
        )
        # checkpoint can be a string (e.g. "best_accuracy" for initializing from a model saved by BestModelLogger)
        # or dictionary with epoch/update/sample values
        if isinstance(initializer_config.checkpoint, str):
            self.checkpoint = initializer_config.checkpoint
        else:
            if not isinstance(initializer_config.checkpoint, dict):
                raise ValueError("checkpoint must be either a string or a dictionary")
            self.checkpoint = TrainingIteration(**initializer_config.checkpoint)
            if not self.checkpoint.is_minimally_specified and not self.checkpoint.is_fully_specified:
                raise ValueError("checkpoint dictionary must be minimally or fully specified")

    def _get_model_state_dict(
        self, model: ModelBase, model_name: str | None = None
    ) -> tuple[dict[str, Tensor], str, Path]:
        """Get the model state dict from the checkpoint.

        Args:
            model: the model to load the state dict into.
            model_name: the name of the model to load.

        Returns:
            state_dict: the model state dict.
            model_name: the name of the model to load.
            checkpoint_uri: the URI of the checkpoint file.
        """
        model_name, checkpoint_uri = self._get_modelname_and_checkpoint_uri(
            model=model, model_name=model_name, file_type="model"
        )
        checkpoint = torch.load(checkpoint_uri, map_location=model.device, weights_only=True)

        if CheckpointKeys.STATE_DICT not in checkpoint:
            raise KeyError(f"Checkpoint at {checkpoint_uri} does not contain a state dict")

        state_dict = checkpoint[CheckpointKeys.STATE_DICT]

        return state_dict, model_name, checkpoint_uri

    def init_optimizer(self, model: ModelBase) -> None:
        """Initialize the optimizer for the model if it is derived from Model.

        If model is a `CompositeModel`, nothing happens. This is expected as CompositeModels can be arbitrarily nested
        and do not have an optimizer. Instead, a CompositeModel calls `init_optim` with all its submodels which can be
        of type `Model` or a nested `CompositeModel`.

        Args:
            model: a model to initialize the optimizer for. Assumes the model has an attribute optim.
        """
        if not isinstance(model, Model):
            return
        if not self.load_optim:
            return

        if model.optimizer is None:
            raise ValueError("Model does not have an optimizer to load state into")

        model_name, ckpt_uri = self._get_modelname_and_checkpoint_uri(model=model, file_type="optim")
        state_dict = torch.load(ckpt_uri, map_location=model.device)
        model.optimizer.load_state_dict(state_dict)
        self.logger.info(f"loaded optimizer of {model_name} from {ckpt_uri}")

    def _get_modelname_and_checkpoint_uri(
        self,
        file_type: Literal["model", "optim"],
        model: ModelBase | None = None,
        model_name: str | None = None,
    ) -> tuple[str, Path]:
        """Get the model name and checkpoint URI.

        Args:
            file_type: a string indicating the type of file to load. "model" for the checkpoint file containing
              model weights or "optim" for the checkpoint file containing the optimizer state.
            model: An instance of the model class from which we read model.name if model_name is not provided and
                self.model_name also not exists.
            model_name: The model name to use.

        Returns:
            model_name: the name of the model to load.
            ckpt_uri: the URI of the checkpoint file.
        """

        if model is None and model_name is None:
            raise ValueError("Either model or model_name must be provided")

        if file_type not in {"model", "optim"}:
            raise ValueError(f"file_type must be 'model' or 'optim', got {file_type}")

        model_name = model_name or self.model_name
        if model_name is None:
            if not isinstance(model, ModelBase):
                raise ValueError("model must be provided if model_name is not set")
            self.logger.info(f"no model_name provided -> using {model.name}")
            model_name = model.name

        # model_info is e.g. ema=0.99
        model_info_str = "" if self.model_info is None else f"_{self.model_info}"

        checkpoint_uri = self._get_checkpoint_uri(prefix=f"{model_name}{model_info_str}_cp=", suffix=f"_{file_type}.th")
        if not checkpoint_uri.exists():
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_uri}' does not exist")
        return model_name, checkpoint_uri

    def _get_checkpoint_uri(self, prefix: str, suffix: str) -> Path:
        """Get the full checkpoint path.

        The checkpoint folder path is inferred from run_id and optionally stage_name.
        The exact checkpoint filename is then inferred from the checkpoint name and the provided prefix and suffix.

        Args:
            prefix: prefix to the checkpoint filename.
            suffix: suffix to the checkpoint filename.

        Returns:
            ckpt_folder / f"{prefix}{ckpt}{suffix}": the full checkpoint path.
        """

        if type(prefix) is not str or type(suffix) is not str:
            raise ValueError("prefix and suffix must be strings")

        checkpoint_path = self.init_run_path_provider.checkpoint_path

        # find full checkpoint from minimal specification
        checkpoint = self.checkpoint
        if not isinstance(self.checkpoint, str) and not self.checkpoint.is_fully_specified:
            checkpoint = TrainingIteration.to_fully_specified_from_filenames(
                directory=checkpoint_path.as_posix(),
                training_iteration=self.checkpoint,
                prefix=prefix,
                suffix=suffix,
            )

        return checkpoint_path / f"{prefix}{checkpoint}{suffix}"
