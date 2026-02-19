#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.callbacks.periodic import IntervalType, PeriodicCallback
from noether.core.schemas.callbacks import CheckpointCallbackConfig
from noether.core.utils.logging import short_number_str
from noether.core.utils.training import UpdateCounter  # fixme: consider moving it to noether/training/?


class CheckpointCallback(PeriodicCallback):
    """Callback to save the model and optimizer state periodically.

    Example config:

    .. code-block:: yaml

        - kind: noether.core.callbacks.CheckpointCallback
          name: CheckpointCallback
          every_n_epochs: 1
          save_weights: true
          save_optim: true
    """

    def __init__(
        self,
        callback_config: CheckpointCallbackConfig,
        **kwargs,
    ):
        """

        Args:
            callback_config: Configuration for the callback. See
                :class:`~noether.core.schemas.callbacks.CheckpointCallbackConfig`
                for available options.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(callback_config=callback_config, **kwargs)
        if not (
            callback_config.save_weights
            or callback_config.save_latest_weights
            or callback_config.save_optim
            or callback_config.save_latest_optim
        ):
            raise ValueError(
                "At least one of save_weights, save_latest_weights, save_optim, save_latest_optim must be True."
            )
        self.save_weights = callback_config.save_weights
        self.save_optim = callback_config.save_optim
        self.save_latest_weights = callback_config.save_latest_weights
        self.save_latest_optim = callback_config.save_latest_optim
        self.model_names = callback_config.model_names

    def before_training(self, *, update_counter: UpdateCounter) -> None:
        frozen_count = self.model.frozen_param_count
        trainable_count = self.model.trainable_param_count

        weight_bytes = (frozen_count + trainable_count) * 4

        # (not 100% accurate...multiple intervals are not considered)
        num_checkpoints = 1
        if self.every_n_epochs is not None and update_counter.end_iteration.epoch is not None:
            num_checkpoints += update_counter.end_iteration.epoch // self.every_n_epochs
        if self.every_n_updates is not None and update_counter.end_iteration.update is not None:
            num_checkpoints += int(update_counter.end_iteration.update / self.every_n_updates)
        if self.every_n_samples is not None and update_counter.end_iteration.sample is not None:
            num_checkpoints += int(update_counter.end_iteration.sample / self.every_n_samples)
        multiplier = 0
        if self.save_weights:
            multiplier += 1
        if self.save_optim:
            multiplier += 2

        if multiplier > 0:
            checkpoint_size_str = (
                f"all {num_checkpoints} checkpoints: "
                + f"{short_number_str(num_checkpoints * weight_bytes * multiplier)}B"
            )
        else:
            checkpoint_size_str = (
                "only latest weights/optim saved."
                if (self.save_latest_weights or self.save_latest_optim)
                else "nothing to save."
            )

        self.logger.info(
            f"Estimated sizes: checkpoint: {short_number_str(weight_bytes * 3)}B, "
            f"weights: {short_number_str(weight_bytes)}B, "
            f"optimizer: {short_number_str(weight_bytes * 2)}B, "
            f"{checkpoint_size_str}"
        )

    def periodic_callback(self, *, interval_type: IntervalType, update_counter: UpdateCounter, **kwargs) -> None:
        if interval_type == "eval":
            return  # checkpoints are only saved during training
        self.checkpoint_writer.save(
            model=self.model,
            trainer=self.trainer,
            checkpoint_tag=str(update_counter.cur_iteration),
            save_weights=self.save_weights,
            save_optim=self.save_optim,
            save_latest_weights=self.save_latest_weights,
            save_latest_optim=self.save_latest_optim,
            model_names_to_save=self.model_names,
            save_frozen_weights=True,
        )

    def after_training(self, **_) -> None:
        self.checkpoint_writer.save(
            model=self.model,
            trainer=self.trainer,
            checkpoint_tag="last",
            save_weights=self.save_weights,
            save_optim=self.save_optim,
            save_latest_weights=self.save_latest_weights,
            save_latest_optim=self.save_latest_optim,
            model_names_to_save=self.model_names,
            save_frozen_weights=True,
        )
