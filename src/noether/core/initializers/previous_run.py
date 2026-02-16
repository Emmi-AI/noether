#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.initializers import CheckpointInitializer
from noether.core.models import CompositeModel, Model, ModelBase
from noether.core.schemas.initializers import PreviousRunInitializerConfig
from noether.core.utils.model import compute_model_norm


class PreviousRunInitializer(CheckpointInitializer):
    """Initializes a model from a checkpoint of a previous run (specified by the run_id), this initializers hence only loads model weights.
    When a previous run should be resumed for further training, use ResumeInitializer instead.
    This initializer needs to be initialized as part of a model config.
    It is possible to remove certain keys or patterns from the checkpoint before loading it into the model, or to rename certain patterns.

    For example:

    .. code-block:: yaml
        model:
          kind: path.to.MyModelClass
          param1: value1
          name: my_model
          initializers:
            - kind: noether.core.initializers.PreviousRunInitializer
            run_id: <run_id>
            model_name: transformer
            stage_name: train
            checkpoint: last
            keys_to_remove:
                - encoder.block1.weight
    """

    def __init__(
        self,
        initializer_config: PreviousRunInitializerConfig,
        **kwargs: dict,
    ):
        """
        Args:
            initializer_config: Configuration for the initializer. See :class:`~noether.core.schemas.initializers.PreviousRunInitializerConfig` for available options.
            **kwargs: additional arguments to pass to the parent class.
        """
        super().__init__(initializer_config=initializer_config, **kwargs)
        self.keys_to_remove = initializer_config.keys_to_remove or []
        self.patterns_to_remove = initializer_config.patterns_to_remove or []
        self.patterns_to_rename = initializer_config.patterns_to_rename or []
        self.patterns_to_instantiate = initializer_config.patterns_to_instantiate or []

    def _init_weights(self, model: ModelBase, model_name: str | None = None) -> None:
        state_dict, model_name, checkpoint_uri = self._get_model_state_dict(model, model_name=model_name)
        if len(self.keys_to_remove) > 0:
            self.logger.info(f"removing keys {self.keys_to_remove} from {checkpoint_uri}")
            for key in self.keys_to_remove:
                state_dict.pop(key)
        if len(self.patterns_to_remove) > 0:
            for pattern in self.patterns_to_remove:
                self.logger.info(f"removing pattern {pattern} from {checkpoint_uri}")
                for key in list(state_dict.keys()):
                    if pattern in key:
                        self.logger.info(f"removing key {key}")
                        state_dict.pop(key)
        if len(self.patterns_to_rename) > 0:
            for rename_pattern in self.patterns_to_rename:
                src_pattern = rename_pattern["src"]
                dst_pattern = rename_pattern.get("dst", "")
                self.logger.info(f"renaming pattern {src_pattern} to {dst_pattern} in {checkpoint_uri}")
                for key in list(state_dict.keys()):
                    if src_pattern in key:
                        new_value = state_dict.pop(key)
                        dst_key = key.replace(src_pattern, dst_pattern)
                        if dst_key in state_dict:
                            self.logger.info(f"overwriting key {dst_key} with {key}")
                        else:
                            self.logger.info(f"renaming key {key} to {dst_key}")
                        state_dict[dst_key] = new_value
        if len(self.patterns_to_instantiate) > 0:
            for inst_pattern in self.patterns_to_instantiate:
                cur_sd = model.state_dict()
                for key in list(cur_sd.keys()):
                    if inst_pattern in key:
                        state_dict[key] = cur_sd[key].clone()

        random_model_norm = compute_model_norm(model)
        model.load_state_dict(state_dict)
        if random_model_norm == compute_model_norm(model):
            raise RuntimeError(
                "Model has not been properly initialized with new weights, model weights are still the same."
            )
        self.logger.info(f"loaded weights of {model_name} from {checkpoint_uri}")

    def init_weights(self, model: ModelBase, model_name: str | None = None) -> None:
        """Initialize the model weights from the checkpoint.

        Args:
            model: the model to load the weights into.
        """
        if not isinstance(model, (Model | CompositeModel)):
            raise TypeError(
                f"PreviousRunInitializer can only initialize Model or CompositeModel instances, got {type(model)}"
            )

        if isinstance(model, CompositeModel):
            for submodule_name, submodel in model.submodels.items():
                # recursively initialize submodels. Single models will be initialized.
                self.init_weights(model=submodel, model_name=f"{model.name}.{submodule_name}")
        else:
            self._init_weights(model=model, model_name=model_name)
