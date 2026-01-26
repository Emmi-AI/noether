How to Implement a Custom Trainer
=================================

To implement a custom trainer in **Noether**, you need to create a new class that inherits from the ``BaseTrainer`` class and implements the ``loss_compute`` method to define your custom loss computation logic.

.. code-block:: python

    from noether.training.trainers import BaseTrainer
    from noether.core.schemas.trainers import BaseTrainerConfig

    class CustomTrainerConfig(BaseTrainerConfig):
        # Add any custom configuration parameters here


    class CustomTrainer(BaseTrainer):

        def __init__(self, trainer_config: CustomTrainerConfig, **kwargs):
            super().__init__(trainer_config, **kwargs)


        def loss_compute(
            self, forward_output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            """Compute the loss given model outputs and targets.
            Args:
                forward_output (dict[str, torch.Tensor]): The output from the model's forward pass.
                targets (dict[str, torch.Tensor]): The ground truth targets.

            Returns:
                dict[str, torch.Tensor]: A dictionary containing the computed loss values.
            """
            # Implement your custom loss computation logic here


The default ``train_step`` implementation of the BaseTrainer calls the ``loss_compute`` method to calculate the loss.
Best practice is to return a dictionary of losses from the ``loss_compute`` method, where each key is a loss name and the value is the corresponding loss tensor.
However, you can also return a single tensor, list or tuple, but then there is no propper naming of the individual losses for logging purposes.
Example YAML configuration:

.. code-block:: yaml
    
    trainer:
      kind: path.to.CustomTrainer
      # Custom configuration parameters can be added here
      forward_properties:
        - coordinates
        - features
      target_properties:
        - labels

You have to define the ``forward_properties`` and ``target_properties`` in the trainer configuration to specify which keys from the batch are used for the model's forward pass and which are used as targets for loss computation.
Those keys are used in the ``train_step`` method call the model forward ``model(**forward_batch)`` as key arguments, and to get the targets ``targets_batch`` for the ``loss_compute`` method.
The default ``train_step`` implementation is as follows:

.. code-block:: python

    def train_step(self, batch: dict[str, Tensor], model: torch.nn.Module) -> TrainerResult:
        """Overriding this function is optional. By default, the `train_step` of the model will be called and is
        expected to return a TrainerResult. Trainers can override this method to implement custom training logic.

        Args:
            batch: Batch of data from which the loss is calculated.
            model: Model to use for processing the data.

        Returns:
            TrainerResult dataclass with the loss for backpropagation, (optionally) individual losses if multiple
            losses are used, and (optionally) additional information about the model forward pass that is passed
            to the callbacks (e.g., the logits and targets to calculate a training accuracy in a callback).
        """
        forward_batch, targets_batch = self._split_batch(batch)
        forward_output = model(**forward_batch)
        additional_outputs = None
        losses = self.loss_compute(forward_output=forward_output, targets=targets_batch)

        if isinstance(losses, tuple) and len(losses) == 2:
            losses, additional_outputs = losses

        if isinstance(losses, torch.Tensor):
            return TrainerResult(
                total_loss=losses, additional_outputs=additional_outputs, losses_to_log={"loss": losses}
            )
        elif isinstance(losses, list):
            losses = {f"loss_{i}": loss for i, loss in enumerate(losses)}

        if len(losses) == 0:
            raise ValueError("No losses computed, check your output keys and loss function.")

        return TrainerResult(
            total_loss=sum(losses.values(), start=torch.zeros_like(next(iter(losses.values())))),
            losses_to_log=losses,
            additional_outputs=additional_outputs,
        )

The default ``train_step`` should cover most deep learning use cases, where there is a forward pass first, and next the loss(es) are computed.
However, if your training logic requires more complex steps (e.g., multiple forward passes, custom optimization steps, etc.), you can override the ``train_step`` method to implement your specific training logic.
The only requirement is to return a ``TrainerResult`` dataclass containing the total loss for backpropagation.