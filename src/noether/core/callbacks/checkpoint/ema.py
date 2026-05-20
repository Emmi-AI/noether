#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, Any, Literal

import torch
import torch.distributed as dist
from pydantic import Field

from noether.core.callbacks.base import CallbackBase, CallBackBaseConfig
from noether.core.callbacks.periodic import IntervalType, PeriodicCallback
from noether.core.distributed import is_distributed, is_rank0
from noether.core.factory import Factory
from noether.core.models.base import ModelBase
from noether.core.providers.path import PathProvider
from noether.core.schemas.lib import Discriminated
from noether.core.types import CheckpointKeys
from noether.core.utils.common import select_with_path
from noether.core.utils.training.counter import UpdateCounter
from noether.core.utils.training.training_iteration import TrainingIteration
from noether.core.writers import PrefixedLogWriter


class EmaCallbackConfig(CallBackBaseConfig):
    name: Literal["EmaCallback"] = Field("EmaCallback", frozen=True)

    target_factors: list[float] = Field(...)
    """The factors for the EMA."""
    model_paths: list[str | None] | None = Field(None)
    """The paths to the models to apply the EMA to (i.e., composite_model.encoder/composite_model.decoder, path of the PyTorch nn.Modules in the checkpoint). If None, the EMA is applied to the whole model. When training with a CompositeModel, the paths on the submodules (i.e., 'encoder', 'decoder', etc.) should be provided via this field, otherwise the EMA will be applied to the CompositeModel as a whole which is not possible to restore later on."""
    save_weights: bool = Field(True)
    """Whether to save the EMA weights."""
    save_last_weights: bool = Field(True)
    """Save the weights of the model when training is over (i.e., at the end of training, save the EMA weights)."""
    save_latest_weights: bool = Field(False)
    """Save the latest EMA weights. Note that the latest weights are always overwritten on the next invocation of this callback."""
    eval_callbacks: list[Annotated[Any, Discriminated(CallBackBaseConfig)]] | None = Field(None)
    """Optional nested periodic callbacks to run against EMA weights. Each child retains its own schedule
    (``every_n_epochs`` etc.); the EMA callback swaps its stored EMA parameters into the live model around
    eval-time hooks (``after_epoch``, ``after_update``, ``at_eval``) and restores the live weights on exit.
    Children are dispatched once per ``target_factor`` and their metric keys are automatically prefixed with
    ``ema=<factor>/`` to avoid collisions with live-model metrics. Note: ``before_training`` and
    ``after_training`` are forwarded without swapping, so EMA initialization and the final save see live
    weights."""


class EmaCallback(PeriodicCallback):
    """Callback for exponential moving average (EMA) of model weights.

    In addition to maintaining and checkpointing EMA weights, this callback can optionally own a list of
    child evaluation callbacks via ``eval_callbacks``. At each eval-time hook (``after_epoch``,
    ``after_update``, ``at_eval``) the EMA weights are swapped into the live model, the children are
    dispatched, and the live weights are restored. Children are dispatched once per ``target_factor`` and
    their metric keys are automatically prefixed with ``ema=<factor>/`` to avoid collisions with live-model
    metrics.

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
          eval_callbacks:
            - kind: noether.training.callbacks.OfflineLossCallback
              every_n_epochs: 1
              dataset_key: val
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

        # Build nested eval callbacks (one set per target_factor). Each child gets a PrefixedLogWriter so its
        # metric keys are namespaced as ``ema=<factor>/<original_key>``. Dispatch (including schedule gating)
        # is still handled by each child's own ``after_epoch`` / ``after_update`` / ``at_eval`` methods.
        self.eval_callbacks: dict[float, list[CallbackBase]] = {}
        eval_callback_configs = getattr(callback_config, "eval_callbacks", None)
        if eval_callback_configs:
            base_kwargs = dict(kwargs)
            for target_factor in self.target_factors:
                prefix = f"ema={target_factor}"
                child_kwargs = {
                    **base_kwargs,
                    "log_writer": PrefixedLogWriter(inner=kwargs["log_writer"], prefix=prefix),
                }
                self.eval_callbacks[target_factor] = Factory().create_list(eval_callback_configs, **child_kwargs)

    def get_children(self) -> list[CallbackBase]:
        """Flat list of child eval callbacks (across all ``target_factors``).

        Exposed to the trainer so nested ``PeriodicDataIteratorCallback`` instances have their samplers
        registered on the shared data loader. The EMA callback remains responsible for dispatching lifecycle
        hooks to its children.
        """
        return [cb for cbs in self.eval_callbacks.values() for cb in cbs]

    def _load_ema_checkpoint(self, ema_path, cur_model, model_path, target_factor):
        """Load EMA state from a checkpoint file."""
        sd = torch.load(ema_path, weights_only=True)[CheckpointKeys.STATE_DICT]
        for name, _ in cur_model.named_parameters():
            self.parameters[(model_path, target_factor)][name] = sd[name]
        for name, _ in cur_model.named_buffers():
            self.buffers[model_path][name] = sd[name]

    def _init_ema_from_model(self, cur_model, model_path, target_factor):
        """Initialize EMA state from the current model weights."""
        for name, param in cur_model.named_parameters():
            self.parameters[(model_path, target_factor)][name] = param.clone()
        for name, buffer in cur_model.named_buffers():
            self.buffers[model_path][name] = buffer.clone()

    def resume_from_checkpoint(self, resumption_paths: PathProvider, model) -> None:
        """Resume EMA state from a checkpoint.

        Tries ``cp=latest`` first (written by periodic saves), then ``cp=last`` (written by
        ``after_training``, e.g. on graceful signal interrupt). If neither exists, falls back to
        initializing EMA from the current model weights.

        Args:
            resumption_paths: :class:`~noether.core.providers.path.PathProvider` with paths to checkpoint files.
            model: Model to resume EMA state for.
        """
        logger = logging.getLogger(type(self).__name__)
        for model_path in self.model_paths:
            cur_model = select_with_path(obj=model, path=model_path)
            if not isinstance(cur_model, torch.nn.Module):
                raise ValueError(f"Path {model_path} on model {self.model} did not resolve to a torch.nn.Module")
            if model_path is None:
                model_name_with_path = model.name
            else:
                model_name_with_path = f"{model.name}.{model_path}"
            for target_factor in self.target_factors:
                cp_dir = resumption_paths.checkpoint_path
                candidates = [
                    cp_dir / f"{model_name_with_path}_ema={target_factor}_cp=latest_model.th",
                    cp_dir / f"{model_name_with_path}_ema={target_factor}_cp=last_model.th",
                ]
                loaded = False
                for ema_path in candidates:
                    if ema_path.exists():
                        logger.info(f"Loading EMA checkpoint from {ema_path.name}")
                        self._load_ema_checkpoint(ema_path, cur_model, model_path, target_factor)
                        loaded = True
                        break
                if not loaded:
                    logger.warning(
                        f"No EMA checkpoint found (tried {[p.name for p in candidates]}), "
                        "initializing EMA from current model weights"
                    )
                    self._init_ema_from_model(cur_model, model_path, target_factor)
                logger.debug(f"EMA state for model path '{model_path}' and target_factor={target_factor} initialized")
        self._was_resumed = True

    def before_training(self, **kwargs) -> None:
        if is_rank0() and not self._was_resumed:
            for model_path in self.model_paths:
                cur_model = select_with_path(obj=self.model, path=model_path)
                if not isinstance(cur_model, torch.nn.Module):
                    raise ValueError(f"Path {model_path} on model {self.model} did not resolve to a torch.nn.Module")
                for target_factor in self.target_factors:
                    self._init_ema_from_model(cur_model, model_path, target_factor)

        # Forward to children on all ranks (without swapping: children may initialize dataset readers,
        # samplers, etc., against the live model).
        for children in self.eval_callbacks.values():
            for child in children:
                child.before_training(**kwargs)

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

    @contextmanager
    def _swapped_weights(self, target_factor: float) -> Iterator[None]:
        """Temporarily replace live model params/buffers with EMA values for ``target_factor``.

        EMA state is only maintained on rank 0 (see :meth:`track_after_update_step`). On non-rank-0 ranks
        the local live weights are used as the broadcast destination, and rank 0 broadcasts its EMA-loaded
        params/buffers to every rank so all ranks run evaluation against the same EMA weights.

        The live weights are always restored on exit, even if the inner block raises. Iterates over all
        configured ``model_paths``.
        """
        snapshots: list[tuple[torch.nn.Module, list[torch.Tensor], list[torch.Tensor], str | None]] = []
        for model_path in self.model_paths:
            cur_model = select_with_path(obj=self.model, path=model_path)
            if not isinstance(cur_model, torch.nn.Module):
                raise ValueError(f"Path {model_path} on model {self.model} did not resolve to a torch.nn.Module")
            live_params = [p.detach().clone() for p in cur_model.parameters()]
            live_buffers = [b.detach().clone() for b in cur_model.buffers()]
            snapshots.append((cur_model, live_params, live_buffers, model_path))

        try:
            for cur_model, _live_params, _live_buffers, model_path in snapshots:
                if is_rank0():
                    key = (model_path, target_factor)
                    for name, param in cur_model.named_parameters():
                        param.data.copy_(self.parameters[key][name])
                    for name, buffer in cur_model.named_buffers():
                        buffer.data.copy_(self.buffers[model_path][name])
                if is_distributed():
                    for param in cur_model.parameters():
                        dist.broadcast(param.data, src=0)
                    for buffer in cur_model.buffers():
                        dist.broadcast(buffer.data, src=0)
            yield
        finally:
            for cur_model, live_params, live_buffers, _model_path in snapshots:
                for param, snap in zip(cur_model.parameters(), live_params, strict=True):
                    param.data.copy_(snap)
                for buffer, snap in zip(cur_model.buffers(), live_buffers, strict=True):
                    buffer.data.copy_(snap)

    def _dispatch_to_children(self, hook: str, *, update_counter: UpdateCounter, **kwargs: Any) -> None:
        """Call ``hook`` on every child, wrapped in the EMA weight swap for its target_factor."""
        if not self.eval_callbacks:
            return
        for target_factor, children in self.eval_callbacks.items():
            with self._swapped_weights(target_factor):
                for child in children:
                    getattr(child, hook)(update_counter=update_counter, **kwargs)

    def after_epoch(self, update_counter: UpdateCounter, **kwargs) -> None:
        # Own periodic save (EMA-cadence gated).
        super().after_epoch(update_counter=update_counter, **kwargs)
        # Children run under EMA weights; each child self-gates on its own schedule.
        self._dispatch_to_children("after_epoch", update_counter=update_counter, **kwargs)

    def after_update(self, update_counter: UpdateCounter, **kwargs) -> None:
        super().after_update(update_counter=update_counter, **kwargs)
        self._dispatch_to_children("after_update", update_counter=update_counter, **kwargs)

    def at_eval(self, update_counter: UpdateCounter, **kwargs) -> None:
        # ``super().at_eval`` calls ``periodic_callback(interval_type="eval")`` which short-circuits for
        # this callback (no EMA save during eval). Still call it for consistency / future hooks.
        super().at_eval(update_counter=update_counter, **kwargs)
        self._dispatch_to_children("at_eval", update_counter=update_counter, **kwargs)

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

    def after_training(self, **kwargs) -> None:
        if self.save_last_weights:
            self._save(training_iteration="last", model=self.model)

        # Forward to children on all ranks (without swapping: children may finalize trackers, flush state,
        # etc., and should not observe EMA-swapped weights after training).
        for children in self.eval_callbacks.values():
            for child in children:
                child.after_training(**kwargs)
