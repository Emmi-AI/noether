Noether Development CLI
========================

To make development of the Noether Framework easier, we introduce the `noether-development` CLI that allows you to develop and test individual components of the framework in isolation.
It makes sure that one does not have to set up a full training loop with all the components of the framework to test a single component, e.g., a dataset, model, or callbacks.

However, although modules can be developed in isolation, some modules still depend on each other in the larger framework.

Here is how the dependency graph of the main Noether modules looks like:

- **Dataset** (1): The dataset is responsible for loading the data per sample from disk and is at the root of the dependency graph.

  - **Pipeline** (2): The multi-stage pipeline (i.e., collator) collates the data loaded by the dataset into a batch and applies additional transformations to the data (if applicable). The pipeline depends on the dataset and cannot be configured without one.

    - **Model** (3): Performs the forward pass on the collated batch.

      - **Trainer** (4.1): Manages the training loop, mainly defining the loss computation.
      - **Callbacks** (4.2): For now only PeriodicDataIteratorCallbacks are supported.

This implies that a model (`3`) cannot be developed without a dataset (`1`) and pipeline (`2`), and a trainer (`4.1`) or callbacks (`4.2`) cannot be developed without a model (`3`), etc.
In practice, it would be possible to develop each component fully in isolation by using dummy data.
However, we decided to enforce that the dependencies between the modules are correct and close to the real use case.


Building a Dataset
------------------

We start with building a simple custom dataset that generates random dummy data.
This dataset is also presented in `development/datasets.py`.

.. code-block:: python

    import torch

    from noether.core.schemas.dataset import StandardDatasetConfig
    from noether.data import Dataset, with_normalizers


    class DevelopmentDatasetConfig(StandardDatasetConfig):
        root: str | None = None
        x_dim: int
        y_dim: int
        z_dim: int
        sample_size: int
        num_samples: int


    class DevelopmentDataset(Dataset):
        def __init__(self, dataset_config: DevelopmentDatasetConfig):
            super().__init__(dataset_config=dataset_config)
            self.dataset_config = dataset_config

        def __len__(self):
            return self.dataset_config.num_samples

        @with_normalizers
        def getitem_x(self, idx: int):
            return torch.randn((self.dataset_config.sample_size, self.dataset_config.x_dim))

        @with_normalizers
        def getitem_y(self, idx: int):
            return torch.randn((self.dataset_config.sample_size, self.dataset_config.y_dim))

        @with_normalizers
        def getitem_z(self, idx: int):
            return torch.randn((self.dataset_config.sample_size, self.dataset_config.z_dim))

Now, if we create a file `development_config.yaml` with the following content, we can test the dataset by running the `noether-development` CLI.
You can configure the batch size to be any value, but for testing purposes we set it to `8` (this is not the same batch size as the effective batch size used in the trainer).

.. code-block:: yaml

    config_schema_kind: development.development_schema.DevelopmentSchema
    batch_size: 8
    datasets:
        development_dataset:
            kind: development.datasets.DevelopmentDatase
            x_dim: 3
            y_dim: 2
            z_dim: 1
            sample_size: 10
            num_samples: 1000
            split: train


Via the following command, the `noether-development` CLI will load the dataset and print out the content of a random batch.

.. code-block:: bash

    uv run noether-development --hp configs/development_config.yaml

The output should look something like this:

.. code-block:: bash

    ============================================================
    RAW BATCH SAMPLE CONTENTS (first random sample from the batch)
    ============================================================
    Batch type: <class 'dict'>
    Batch keys (4): ['index', 'x', 'y', 'z']

    [index]
        type: int
        value:  455

    [x]
        type: Tensor
        shape:  (1024, 3)
        dtype:  torch.float32
        device: cpu
        first 10 values: [-1.154662013053894, 0.4905768632888794, ....

You can now test the behavior of the dataset and its normalizers, either via the CLI or via breakpoints in debug mode (via calling the CLI tool).
If you want to add normalizers to the dataset, you can do so by adding the following to the dataset config in `development_config.yaml`:

.. code-block:: yaml

    development_dataset:
        ...
        dataset_normalizers:
            x:
                - kind: noether.data.preprocessors.normalizers.MeanStdNormalization
                  mean: [0, 0, 0]
                  std: [1, 1, 1]
            y:
                - kind: noether.data.preprocessors.normalizers.MeanStdNormalization
                  mean: [1, 1]
                  std: [1, 1]
            z:
                - kind: noether.data.preprocessors.normalizers.MeanStdNormalization
                  mean: [2]
                  std: [1]

Running the CLI again will now print out the normalized batch content, centered around the mean values defined in the normalizers (i.e., `0`, `1`, `2`).

Building a Pipeline
-------------------

Given that we have a dataset that loads and normalizes tensor data, we can continue to build the pipeline that collates the samples into batches.

.. code-block:: python

    from pydantic import Field

    from noether.core.schemas.dataset import PipelineConfig
    from noether.data.pipeline import MultiStagePipeline
    from noether.data.pipeline.collators import (
        DefaultCollator,
    )
    from noether.data.pipeline.sample_processors import (
        ConcatTensorSampleProcessor,
        PointSamplingSampleProcessor,
    )


    class DevelopmentPipelineConfig(PipelineConfig):
        num_points: int = Field(default=56, description="Number of points to sample from the point cloud.")


    class DevelopmentPipeline(MultiStagePipeline):
        def __init__(self, config: DevelopmentPipelineConfig):
            self.config = config

            super().__init__(
                sample_processors=[
                    PointSamplingSampleProcessor(items=["x", "y", "z"], num_points=self.config.num_points),
                    ConcatTensorSampleProcessor(items=["x", "z"], target_key="x_z", dim=1),
                ],
                batch_processors=[],
                collators=[DefaultCollator(items=["x_z", "y"])],
            )

This simple multi-stage pipeline samples the `x`, `y`, `z` tensors to a fixed number of points and concatenates the `x` and `z` tensors into a new tensor called `x_z`, which is then collated together with the `y` tensor into a batch.
To add the pipeline to the development config, we can add the following to the dataset config in `development_config.yaml`:

.. code-block:: yaml

    development_dataset:
        ...
        pipeline:
            kind: development.pipeline.DevelopmentPipeline
            num_points: 64

Now when we run the CLI again, we can see the collated batch content, which should now contain the collated tensors `x_z` and `y`.

.. code-block:: bash

    ============================================================
    COLLATED BATCH CONTENTS
    ============================================================
    Batch type: <class 'dict'>
    Batch keys (2): ['x_z', 'y']

    [x_z]
        type: Tensor
        shape:  (8, 64, 5)
        dtype:  torch.float32
        device: cpu
        first 10 values: [-0.24723079800605774, -0.06603412330150604, -0.47881123423576355, 0.3252887725830078, ...

    [y]
        type: Tensor
        shape:  (8, 64, 2)
        dtype:  torch.float32
        device: cpu
        first 10 values: [-1.2091279029846191, -1.2091279029846191, -0.3098295032978058, -0.3098295032978058, ...

We can see that the collated batch contains the `x_z` (with dimensions `x + z`) and `y` tensors, both containing `8` samples with `64` points, which are the outputs of the pipeline and the collator, respectively.

Building a Model 
----------------

Next we can build a simple model that takes the collated batch as input and performs a forward pass on the `x_z` tensor to predict the `y` tensor.

.. code-block:: python

    import torch

    from noether.core.models import Model
    from noether.core.schemas.models import ModelBaseConfig


    class DevelopmentModelConfig(ModelBaseConfig):
        kind: str = "development.DevelopmentModel"
        name: str = "development_model"
        input_dim: int
        hidden_dim: int = 256
        output_dim: int


    class DevelopmentModel(Model):
        def __init__(self, model_config: DevelopmentModelConfig, **kwargs):
            super().__init__(model_config=model_config, **kwargs)

            self.layer1 = torch.nn.Linear(self.model_config.input_dim, self.model_config.hidden_dim)
            self.layer2 = torch.nn.Linear(self.model_config.hidden_dim, self.model_config.output_dim)

        def forward(self, x_z: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.layer1(x_z))
            x = self.layer2(x)

            return {"output": x}

Matching to the model, we add the following config to the `development_config.yaml`:

.. code-block:: yaml

    ...
    model: # can be commented out to test behavior when no model is defined, skip model instantiation and forward pass
        kind: development.model.DevelopmentModel
        input_dim: 5
        output_dim: ${datasets.development_dataset.y_dim}
        forward_properties: ["x_z"]


If we now run the CLI again, we can see the output of the forward pass of the model on the collated batch content.

.. code-block:: bash

    ============================================================
    MODEL OUTPUT CONTENTS
    ============================================================
    Batch type: <class 'dict'>
    Batch keys (1): ['output']

    [output]
        type: Tensor
        shape:  (8, 64, 1)
        dtype:  torch.float32
        device: cpu
        first 10 values: [-0.19360095262527466, 0.00014679878950119019, -0.21560004353523254, ...
    ============================================================

You can see that we get an output tensor with the same number of samples and points as the input `x_z` tensor, but with a different number of features (i.e., `output_dim`), which is the output of the model forward pass.
Note that it is not necessary to configure an optimizer with the model (which is needed for running a training loop).

Building a Trainer and Callbacks
-------------------------------- 

Finally, we can work on the last two modules that depend on the dataset/pipeline/model, which are the trainer and the callbacks.
We first start with the trainer, where the user has to define the loss computation:

.. code-block:: python


    import torch.nn.functional as F

    from noether.core.schemas.trainers import BaseTrainerConfig
    from noether.training.trainers import BaseTrainer


    class DevelopmentTrainerConfig(BaseTrainerConfig):
        pass


    class DevelopmentTrainer(BaseTrainer):
        def __init__(self, trainer_config: DevelopmentTrainerConfig, **kwargs):
            super().__init__(
                config=trainer_config,
                **kwargs,
            )

            self.config = trainer_config

        def loss_compute(self, forward_output: dict[str, any], targets: dict[str, any]) -> dict[str, any]:
            loss = F.mse_loss(forward_output["output"], targets["y"])
            return {"loss": loss}

Then add the following to the `development_config.yaml` to configure the trainer:

.. code-block:: yaml

    ...
    trainer: # can be commented out to test behavior when no trainer is defined, skip training loop
        kind: development.trainer.DevelopmentTrainer
        effective_batch_size: ${development_batch_size} # has to be set, but not used during development
        max_epochs: 10 # has to be set, but not used during development
        callbacks: null
        forward_properties: ${model.forward_properties}
        target_properties: ["y"]

The model will call the `trainer.train_step` method, which will compute the loss on the forward output of the model and the target `y` tensor from the dataset.
Adding the trainer to the config produces output like this:

.. code-block:: bash

    Backward pass completed successfully.

    All model parameters received gradients successfully.

    ============================================================
    TRAINER OUTPUT
    ============================================================
    total_loss: 1.061652

    losses_to_log (1):
        [loss]: 1.061652

    additional_outputs: None
    ============================================================


The development pipeline performs a backward pass on the loss computed by the trainer, and checks that all model parameters receive gradients successfully.
Additionally, the outputs of `loss_compute` are printed as the trainer output.


Finally, we can add a callback to the trainer (since the callbacks are configured as part of the trainer) to test the behavior of the callbacks in the training loop. For now, it is only possible to add `PeriodicDataIteratorCallbacks` to the trainer during development.

.. code-block:: python


    from typing import Literal

    import torch
    import torch.nn.functional as F

    from noether.core.callbacks.periodic import PeriodicDataIteratorCallback
    from noether.core.schemas.callbacks import PeriodicDataIteratorCallbackConfig


    class DevelopmentCallbackConfig(PeriodicDataIteratorCallbackConfig):
        name: Literal["DevelopmentCallback"] = "DevelopmentCallback"
        forward_properties: list[str] = []


    class DevelopmentCallback(PeriodicDataIteratorCallback):
        def __init__(self, callback_config: DevelopmentCallbackConfig, **kwargs):
            super().__init__(callback_config=callback_config, **kwargs)
            self.config = callback_config

        def process_data(self, batch: dict[str, torch.Tensor], **kwargs) -> None:
            loss = F.mse_loss(
                self.model(**{prop: batch[prop] for prop in self.config.forward_properties})["output"], batch["y"]
            )

            return {"mse_loss": loss.item()}

This simple callback computes the mean squared error loss on the forward output of the model and the target `y` tensor from the dataset, and returns it as an additional output.
To use the callback, we can add the following to the trainer config in `development_config.yaml`:

.. code-block:: yaml

    ...
    trainer:
        ...
        callbacks:
            - kind: development.callbacks.DevelopmentCallback
              batch_size: 1
              every_n_epochs: 1
              dataset_key: development_dataset
              name: DevelopmentCallback
              forward_properties: ${model.forward_properties}


Only the `process_data` method is called in the development pipeline.
The output of the callback is printed in the CLI, which should look something like this:

.. code-block:: bash

    ============================================================
    CALLBACK OUTPUT CONTENTS
    ============================================================
    Batch type: <class 'dict'>
    Batch keys (1): ['mse_loss']

    [mse_loss]
        type: float
        value:  1.2776843309402466
    ============================================================


Running a full training with the development setup
==================================================

Now that we have built all the required components for a full training loop, we can put them to use with `noether-train`.

We need to add an optimization config to the model configuration, for example:

.. code-block:: yaml

    ...
    model:
        ...
        optimizer_config:
            kind: torch.optim.AdamW
            lr: 1.0e-3
            weight_decay: 0.05
            clip_grad_norm: 1.0
            schedule_config:
                kind: noether.core.schedules.LinearWarmupCosineDecaySchedule
                warmup_percent: 0.05
                end_value: 1.0e-6
                max_value: ${model.optimizer_config.lr}

and at the top of the config we need to add:

.. code-block:: bash

    output_path: <path_to_output_dir>

That's it. Now it should be possible to run a full training with the `noether-train` CLI using the config and modules just developed.

.. code-block:: bash

    uv run noether-train --hp configs/development_config.yaml  +seed=1 +devices=\"0\" +accelerator=cpu