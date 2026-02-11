#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import torch
from torch.amp.grad_scaler import GradScaler

from noether.core.schemas.optimizers import OptimizerConfig
from noether.core.utils.bidict import Bidict
from noether.core.utils.logging import float_to_scientific_notation
from noether.core.utils.torch import NoopGradScaler

if TYPE_CHECKING:
    from noether.core.models import Model
    from noether.core.optimizer.param_group_modifiers import ParamGroupModifierBase
    from noether.core.utils.training.counter import UpdateCounter
    from noether.core.utils.training.schedule_wrapper import ScheduleWrapper


class OptimizerWrapper:
    """Wrapper around an `torch.optim.Optimizer` that allows
        - excluding biases and weights of normalization layers from weight decay
        - creating param_groups (e.g., for a layerwise lr scaling)
        - learning rate scheduling
        - gradient clipping
        - weight decay scheduling
    Have a look at the class:`noether.core.schemas.optimizers.OptimizerConfig` for available options.
    """

    schedule: ScheduleWrapper | None = None
    weight_decay_schedule: ScheduleWrapper | None = None

    def __init__(
        self,
        model: Model,
        torch_optim_ctor: Callable[[Iterable[dict[str, Any]]], torch.optim.Optimizer],
        optim_wrapper_config: OptimizerConfig,
        update_counter: UpdateCounter | None = None,
    ):
        """
        Args:
            model: Parameters of this model will be optimized.
            torch_optim_ctor: The :class:`~torch.optim.Optimizer` that should be wrapped. Needs to be a callable
                because it requires the parameters of the model for initialization.
            optim_wrapper_config: The configuration for the optimizer wrapper.
            update_counter: Object that provides the current training progress to enable scheduling of the learning
                rate or the weight decay.
        """
        # import here to avoid circular dependency
        from noether.core.factory import Factory, ScheduleFactory

        self.logger = logging.getLogger(type(self).__name__)

        self.model = model
        self.update_counter = update_counter
        self.config = optim_wrapper_config

        # create a param group for each parameter
        param_group_modifiers = Factory().create_list(self.config.param_group_modifiers_config)
        self.logger.debug(
            f"exclude_bias_from_wd={self.config.exclude_bias_from_weight_decay} exclude_norm_from_wd={self.config.exclude_normalization_params_from_weight_decay} "
            f"param_group_modifiers=[{' '.join(str(pgm) for pgm in param_group_modifiers)}]"
        )
        # apply param group modifiers to the named parameters of the model
        param_groups = self._apply_parameter_modifiers(model, param_group_modifiers)
        # merge groups with same properties (useful for logging)
        merged_groups, merged_groups_paramnames = self._merge_groups_with_the_same_parameters(param_groups)
        merged_groups = self._add_names_to_param_groups(merged_groups)
        self._log_param_groups(merged_groups, merged_groups_paramnames)

        # torch optimizer organizes parameters by enumerating them (not by name)
        # so for loading an arbitrary optim state_dict an association from param_name to param_idx has to be stored
        self.param_idx_to_name = Bidict[int, str]()
        idx = 0
        for group_paramnames in merged_groups_paramnames:
            for param_name in group_paramnames:
                self.param_idx_to_name.set(idx, param_name)
                idx += 1

        # initialize torch optim
        self.torch_optim = torch_optim_ctor(merged_groups)

        # for grad clipping all parameters of the model are required
        self.all_parameters = None
        if self.config.clip_grad_value is not None or self.config.clip_grad_norm is not None:
            self.all_parameters = list(model.parameters())

        self._apply_learning_rate_scaling()

        # create schedules
        if optim_wrapper_config.schedule_config is not None:
            self.schedule = ScheduleFactory().create(
                optim_wrapper_config.schedule_config,
                update_counter=self.update_counter,
            )
        if optim_wrapper_config.weight_decay_schedule is not None:
            self.weight_decay_schedule = ScheduleFactory().create(
                optim_wrapper_config.weight_decay_schedule,
                update_counter=self.update_counter,
            )
            # store initial_lr/initial_wd in param_groups
            # NOTE: torch optimizer broadcasts all values to all param groups (so every param_group has a weight_decay)
            for param_group in self.torch_optim.param_groups:
                assert "exclude_from_wd" not in param_group
                param_group["exclude_from_wd"] = param_group["weight_decay"] == 0.0

    def _apply_learning_rate_scaling(self):
        """Applies the learning rate scaling to the parameter groups in the optimizer."""
        for param_group in self.torch_optim.param_groups:
            if "lr_scale" in param_group:
                if "original_lr" in param_group:
                    raise ValueError("original_lr is already in param_group, this should not happen")

                param_group["original_lr"] = param_group["lr"]
                # lr is float so inplace operation is fine
                # this scaling is only relevant for logging and epoch based schedules
                # for update based schedule the value is anyway scaled again at the start of the update
                param_group["lr"] *= param_group["lr_scale"]
                self.logger.info(
                    f"scaled lr of param_group '{param_group['name']}' "
                    f"from {float_to_scientific_notation(param_group['original_lr'], max_precision=2)} "
                    f"to {float_to_scientific_notation(param_group['lr'], max_precision=2)}"
                )

    def _log_param_groups(self, merged_groups, merged_groups_paramnames) -> None:
        def param_group_str(param_group):
            return " ".join(
                [f"{key}={value}" for key, value in param_group.items() if key not in ["params", "name"]]
                or ["default"] + [f"len(params)={len(param_group['params'])}"]
            )

        param_group_names = ", ".join([param_group_str(pg) for pg in merged_groups])
        self.logger.info(f"Using {len(merged_groups)} param groups: {param_group_names}")
        for i in range(len(merged_groups)):
            for param_name in merged_groups_paramnames[i]:
                self.logger.debug(f"- {param_name}")

    def _apply_parameter_modifiers(
        self, model: Model, param_group_modifiers: list[ParamGroupModifierBase]
    ) -> list[dict[str, Any]]:
        """Applies the parameter group modifiers to parameters of the model with the same name.

        Args:
            model: The model for which the parameters should be optimized.
            param_group_modifiers: The parameter group modifiers that should be applied.
        """

        param_groups = []
        for name, param in model.named_parameters():
            properties: dict[str, Any] = {}
            # excluding norm and bias params is very common for all models -> support with simple flag
            # bias has ndim == 1, so it needs to be checked before
            # the bias of norm layers is considered a bias, not a norm parameter
            if name.split(".")[-1] == "bias" and self.config.exclude_bias_from_weight_decay:
                properties["weight_decay"] = 0.0
            # timm does it like this...not sure if other parameters can also have ndim <= 1
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/optim_factory.py
            elif (
                param.ndim <= 1
                and self.config.exclude_normalization_params_from_weight_decay
                and name.split(".")[-1] != "bias"
            ):
                properties["weight_decay"] = 0.0

            for param_group_modifier in param_group_modifiers:
                for key, value in param_group_modifier.get_properties(model, name, param).items():
                    if key in properties and key == "lr_scale":
                        properties[key] *= value
                    else:
                        if key in properties:
                            raise ValueError(f"{key} is already in {properties}")
                        properties[key] = value
            if "params" in properties:
                raise ValueError("param group modifier returned 'params' as property, which is not allowed")
            if "name" in properties:
                raise ValueError("param group modifier returned 'name' as property, which is not allowed")

            properties["name"] = name
            properties["params"] = [param]
            param_groups.append(properties)

        # check that param group modifiers were successfully applied (e.g. check that param name was found in model)
        for param_group_modifier in param_group_modifiers:
            if not param_group_modifier.was_applied_successfully():
                raise ValueError(f"{param_group_modifier} failed")

        return param_groups

    def _add_names_to_param_groups(self, merged_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Adds a name to each param group for better logging."""

        for param_group in merged_groups:
            names = []
            for key, value in param_group.items():
                if key == "params":
                    continue
                if isinstance(value, float):
                    if value == 0.0:
                        value_str = "0"
                    else:
                        value_str = float_to_scientific_notation(value, max_precision=1, remove_plus=True)
                elif isinstance(value, int):
                    value_str = str(value)
                else:
                    raise NotImplementedError
                names.append(f"{key}={value_str}")
            if len(names) == 0:
                param_group["name"] = "default"
            else:
                param_group["name"] = "&".join(names)
        return merged_groups

    def _merge_groups_with_the_same_parameters(self, param_groups) -> tuple[list[dict[str, Any]], list[list[str]]]:
        """
        Merges parameter groups with the same properties.

        Args:
            param_groups: The parameter groups to merge. {'params': [...], name: 'parameter_name', weight_decay': 0.01, 'lr_scale': 0.5, }.
            I.e., merge by lr_scale = 0.5 and weight_decay = 0.01, if multiple named parameters have the same properties.
        """
        merged_groups: list[dict[str, Any]] = []
        merged_groups_properties: list[dict[str, Any]] = []
        merged_groups_paramnames: list[list[str]] = []
        for param_group in param_groups:
            param_name = param_group.pop("name")
            properties = {k: v for k, v in param_group.items() if k != "params"}
            matching_group_idx = None
            for i, merged_group_properties in enumerate(merged_groups_properties):
                if properties == merged_group_properties:
                    matching_group_idx = i
                    break
            if matching_group_idx is None:
                merged_groups.append(param_group)
                merged_groups_properties.append(properties)
                merged_groups_paramnames.append([param_name])
            else:
                merged_groups[matching_group_idx]["params"] += param_group["params"]
                merged_groups_paramnames[matching_group_idx].append(param_name)
        return merged_groups, merged_groups_paramnames

    def _has_param_with_grad(self) -> bool:
        """Checks if any parameter of the optimizer requires a gradient."""
        for param_group in self.torch_optim.param_groups:
            for p in param_group["params"]:
                if p.grad is not None:
                    return True
        return False

    def step(self, grad_scaler: GradScaler | None = None) -> None:
        """Wrapper around `torch.optim.Optimizer.step` which automatically handles:
        - gradient scaling for mixed precision (including updating the GradientScaler state)
        - gradient clipping
        - calling the .step function of the optimizer
        """
        # grad_scaler doesnt support update without gradient (e.g. GAN setting)
        # Error: AssertionError: No inf checks were recorded for this optimizer
        if isinstance(grad_scaler, GradScaler):
            if not self._has_param_with_grad():
                return

        # grad scaler is not strictly required
        # (e.g. if OptimizerWrapper is only used for excluding bias/norm parameters from weight decay)
        if grad_scaler is None:
            grad_scaler = NoopGradScaler()

        # NOTE: closure is not supported with GradScaler
        if self.config.clip_grad_value is not None or self.config.clip_grad_norm is not None:
            grad_scaler.unscale_(self.torch_optim)
        # clip gradients
        if self.config.clip_grad_value is not None:
            torch.nn.utils.clip_grad_value_(self.all_parameters, self.config.clip_grad_value)
        if self.config.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.all_parameters, self.config.clip_grad_norm)
        # torch optim step with grad scaler
        grad_scaler.step(self.torch_optim)
        grad_scaler.update()

    def schedule_step(self) -> None:
        """Applies the current state of the schedules to the parameter groups."""
        if self.schedule is not None:
            lr_scale = self.schedule.get_value()
            for param_group in self.torch_optim.param_groups:
                if "lr_scale" in param_group:
                    # lr_scale -> current lr from schedule
                    # param_group["lr_scale"] -> scale form layer-wise lr decay
                    param_group["lr"] = param_group["lr_scale"] * lr_scale
                else:
                    param_group["lr"] = lr_scale
        if self.weight_decay_schedule is not None:
            wd_scale = self.weight_decay_schedule.get_value()
            for param_group in self.torch_optim.param_groups:
                if not param_group["exclude_from_wd"]:
                    param_group["weight_decay"] = wd_scale

    def zero_grad(self, set_to_none=True):
        """Wrapper around `torch.optim.Optimizer.zero_grad`."""
        # set_to_none is True by default (unlike torch.optim.optimizer)
        self.torch_optim.zero_grad(set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Wrapper around `torch.optim.Optimizer.state_dict`. Additionally adds info about index to name mapping."""
        sd = self.torch_optim.state_dict()
        sd["param_idx_to_name"] = self.param_idx_to_name.to_forward()
        return sd  # type: ignore[no-any-return]

    def load_state_dict(self, state_dict_to_load: dict[str, Any]) -> None:
        """Wrapper around `torch.optim.Optimizer.load_state_dict`. Additionally handles edge cases if the parameter
        groups of the loaded state_dict do not match the current configuration. By default, torch would overwrite the
        current parameter groups with the one from the checkpoint. This is undesireable in the following cases:
        - add new parameters (e.g. unfreeze something)
        - change weight_decay or other param_group properties: the load_state_dict would overwrite the actual
            weight_decay (defined in the constructor of the OptimizerWrapper) with the weight_decay from the checkpoint

        Args:
             state_dict_to_load: The optimizer state to load.
        """
        # torch optim state_dict stores param_groups and the states of each parameter
        # if a torch optim state_dict is loaded it would overwrite all param_groups from the checkpoint
        # this results in unexpected behavior when loading an optimizer (e.g. for resuming a run from a checkpoint)
        # - add new parameters (e.g. unfreeze something)
        # - change weight_decay or other param_group properties: the load_state_dict would overwrite the actual
        #   weight_decay with the weight_decay from the checkpoint
        if "param_idx_to_name" in state_dict_to_load:
            # torch optim stores:
            # - a list of param_idxs in each param_group
            # - a dict from param_idxs to state for the state of the param
            # -> match the param_idxs and overwrite the state
            loaded_param_idx_to_name = Bidict(forward=state_dict_to_load["param_idx_to_name"])
            loaded_states = state_dict_to_load["state"]
            cur_state_dict = self.torch_optim.state_dict()
            cur_states = cur_state_dict["state"]
            cur_param_groups = cur_state_dict["param_groups"]
            for cur_param_group in cur_param_groups:
                for cur_param_idx in cur_param_group["params"]:
                    param_name = self.param_idx_to_name.get_value_by_key(cur_param_idx)
                    loaded_param_idx = loaded_param_idx_to_name.get_key_by_value(param_name)
                    if loaded_param_idx not in loaded_states:
                        # if no optim step was done no state exists -> dont load the state
                        cur_states.pop(loaded_param_idx, None)
                    else:
                        # overwrite state with loaded state
                        cur_states[cur_param_idx] = loaded_states[loaded_param_idx]
            state_dict_to_load = dict(state=cur_states, param_groups=cur_param_groups)
        self.torch_optim.load_state_dict(state_dict_to_load)

    def __repr__(self) -> str:
        return f"OptimizerWrapper({self.torch_optim})"

    def __str__(self) -> str:
        s = f"{self.torch_optim.__class__.__name__}"
        attrs = self.config.model_dump(exclude_defaults=True, exclude_none=True, exclude_unset=True)
        if len(attrs) > 0:
            s += "("
            s += ", ".join(f"{key}={value}" for key, value in attrs.items())
            s += ")"
        return s
