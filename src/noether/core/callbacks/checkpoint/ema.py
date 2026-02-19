#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections import defaultdict

import torch

from noether.core.callbacks.periodic import IntervalType, PeriodicCallback
from noether.core.distributed import is_rank0
from noether.core.models.base import ModelBase
from noether.core.providers.path import PathProvider
from noether.core.schemas.callbacks import EmaCallbackConfig
from noether.core.types import CheckpointKeys
from noether.core.utils.common import select_with_path
from noether.core.utils.training.training_iteration import TrainingIteration


class EmaCallback(PeriodicCallback):
    """Callback for exponential moving average (EMA) of model weights.

    Example config:

    .. code-block:: yaml

        - kind: noether.core.callbacks.EmaCallback
          every_n_epochs: 10
          save_weights: false
          save_last_weights: false
          save_latest_weights: true
          target_factors:
            - 0.9999
          name: EmaCallback
    """

    def __init__(
        self,
        callback_config: EmaCallbackConfig,
        **kwargs,
    ):
        """

        Args:
            callback_config: Configuration for the callback. See
                :class:`~noether.core.schemas.callbacks.EmaCallbackConfig`
                for available options.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(callback_config=callback_config, **kwargs)
        self.model_paths = callback_config.model_paths or [None]
        self.target_factors = callback_config.target_factors
        self.save_weights = callback_config.save_weights
        self.save_last_weights = callback_config.save_last_weights
        self.save_latest_weights = callback_config.save_latest_weights
        self.parameters: dict[tuple[str | None, float], dict[str, torch.Tensor]] = defaultdict(dict)
        self.buffers: dict[str | None, dict[str, torch.Tensor]] = defaultdict(dict)
        self._was_resumed = False

    def resume_from_checkpoint(self, resumption_paths: PathProvider, model) -> None:
        """Resume EMA state from a checkpoint.

        Args:
            resumption_paths: :class:`~noether.core.providers.path.PathProvider` with paths to checkpoint files.
            model: Model to resume EMA state for.
        """
        for model_path in self.model_paths:
            cur_model = select_with_path(obj=model, path=model_path)
            if not isinstance(cur_model, torch.nn.Module):
                raise ValueError(f"Path {model_path} on model {self.model} did not resolve to a torch.nn.Module")
            if model_path is None:
                model_name_with_path = model.name
            else:
                model_name_with_path = f"{model.name}.{model_path}"
            for target_factor in self.target_factors:
                sd = torch.load(
                    resumption_paths.checkpoint_path / f"{model_name_with_path}_ema={target_factor}_cp=latest_model.th",
                    weights_only=True,
                )[CheckpointKeys.STATE_DICT]
                for name, _ in cur_model.named_parameters():
                    self.parameters[(model_path, target_factor)][name] = sd[name]
                for name, _ in cur_model.named_buffers():
                    self.buffers[model_path][name] = sd[name]
        self._was_resumed = True

    def before_training(self, **_) -> None:
        if not is_rank0():
            return
        if self._was_resumed:
            return
        for model_path in self.model_paths:
            cur_model = select_with_path(obj=self.model, path=model_path)
            if not isinstance(cur_model, torch.nn.Module):
                raise ValueError(f"Path {model_path} on model {self.model} did not resolve to a torch.nn.Module")
            for target_factor in self.target_factors:
                for name, param in cur_model.named_parameters():
                    self.parameters[(model_path, target_factor)][name] = param.clone()
            for name, buffer in cur_model.named_buffers():
                self.buffers[model_path][name] = buffer.clone()

    def apply_ema(self, cur_model, model_path, target_factor):
        """fused in-place implementation"""
        key = (model_path, target_factor)
        target_param_list = list(self.parameters[key].values())
        source_param_list = list(cur_model.parameters())
        # noinspection PyProtectedMember
        torch._foreach_mul_(target_param_list, target_factor)
        # noinspection PyProtectedMember
        torch._foreach_add_(target_param_list, source_param_list, alpha=1 - target_factor)

    def track_after_update_step(self, **_) -> None:
        if not is_rank0():
            return

        for model_path in self.model_paths:
            cur_model = select_with_path(obj=self.model, path=model_path)

            if not isinstance(cur_model, torch.nn.Module):
                raise ValueError(f"Path {model_path} on model {self.model} did not resolve to a torch.nn.Module")

            for target_factor in self.target_factors:
                self.apply_ema(cur_model, model_path, target_factor)

            for name, buffer in cur_model.named_buffers():
                self.buffers[model_path][name].copy_(buffer)

    def _save(self, training_iteration: str | TrainingIteration, model: ModelBase) -> None:
        if not is_rank0():
            return

        training_iteration_str = str(training_iteration)

        for model_path in self.model_paths:
            cur_model = select_with_path(obj=model, path=model_path)
            if not isinstance(cur_model, ModelBase):
                raise ValueError(f"Path {model_path} on model {self.model} did not resolve to a ModelBase")

            cur_model_path = model.name if model_path is None else f"{model.name}.{model_path}"

            for target_factor in self.target_factors:
                state_dict = {**self.parameters[(model_path, target_factor)], **self.buffers[model_path]}

                save_requests = [
                    (self.save_weights, training_iteration_str),
                    (self.save_latest_weights, "latest"),
                ]

                for should_save, checkpoint in save_requests:
                    if not should_save:
                        continue
                    self.checkpoint_writer.save_model_checkpoint(
                        model_name=cur_model_path,
                        checkpoint_tag=checkpoint,
                        model_info=f"ema={target_factor}",
                        state_dict=state_dict,
                        model_config=getattr(model, "model_config", None),
                        ema=target_factor,
                    )

    def periodic_callback(self, *, interval_type: IntervalType, update_counter, **_) -> None:
        if interval_type == "eval":
            return  # checkpoints are only saved during training
        checkpoint = update_counter.cur_iteration
        self._save(checkpoint, model=self.model)

    def after_training(self, **_) -> None:
        if self.save_last_weights:
            self._save(training_iteration="last", model=self.model)
