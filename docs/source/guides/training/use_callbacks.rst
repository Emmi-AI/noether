How to Use and Implement Callbacks
==================================

Callbacks are the primary mechanism in the **Noether Framework** that allow you to inject custom logic into various stages of the training process. They are primarily used for monitoring, checkpointing, evaluation, and experiment tracking.

What are Callbacks?
-------------------

In Noether, a callback is a class that inherits from :class:`~noether.core.callbacks.base.CallbackBase`. These objects provide hooks that the trainer calls at specific points:

*   **Before training starts**: Initialization, logging hyperparameters, or printing model summaries.
*   **After each accumulation step**: Tracking metrics across multiple batches.
*   **After each optimizer update**: Updating learning rate schedules or tracking update-level metrics.
*   **After each epoch**: Performing validation or saving periodic checkpoints.
*   **After training ends**: Final evaluation, saving final results, or cleanup.
*   **At evaluation time**: Running inference on validation or test datasets.

Types of Callbacks
------------------

Noether provides several base classes and a wide range of built-in callbacks.


Pre/Post Training Callbacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These callbacks inherit directly from :class:`~noether.core.callbacks.base.CallbackBase` and are designed to run logic only once at the start or end of training. Examples include logging hyperparameters before training or saving final results after training.

Periodic Callbacks
~~~~~~~~~~~~~~~~~~

Most callbacks during training are **periodic**. Inheriting from :class:`~noether.core.callbacks.periodic.PeriodicCallback` allows you to configure how often a callback should run using one of the following intervals:

*   ``every_n_epochs``: Runs after every $N$ epochs.
*   ``every_n_updates``: Runs after every $N$ optimizer steps.
*   ``every_n_samples``: Runs after every $N$ training samples have been processed.


Periodic Data Iterator Callbacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For tasks that require iterating over an entire dataset (like validation or computing complex metrics on a test set), Noether provides :class:`~noether.core.callbacks.periodic.PeriodicDataIteratorCallback`. 

This class handles:

*   Distributed data sampling across multiple GPUs.
*   Automatic collation of results from different ranks.
*   Integration with the training data pipeline.

Commonly Used Callbacks
-----------------------

Noether includes many pre-defined callbacks organized by their purpose:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Category
     - Examples and Usage
   * - **Monitoring**
     - :class:`~noether.core.callbacks.default.ProgressCallback`, :class:`~noether.core.callbacks.default.DatasetStatsCallback`, :class:`~noether.core.callbacks.default.LrCallback`, :class:`~noether.core.callbacks.default.PeakMemoryCallback`, :class:`~noether.core.callbacks.default.OnlineLossCallback`, :class:`~noether.core.callbacks.default.ParamCountCallback`, :class:`~noether.core.callbacks.default.EtaCallback`, :class:`~noether.core.callbacks.default.TrainTimeCallback`. Used for real-time tracking of training progress and hardware usage. These callbacks are all initialized by default by the :class:`~noether.training.trainers.BaseTrainer`, the user does not need to add them manually.
   * - **Checkpointing**
     - :class:`~noether.core.callbacks.BestCheckpointCallback`, :class:`~noether.core.callbacks.CheckpointCallback`, :class:`~noether.core.callbacks.EMACallback`. Used to save model weights periodically or when a new best metric is achieved.
   * - **Early Stopping**
     - :class:`~noether.core.callbacks.MetricEarlyStopping`, :class:`~noether.core.callbacks.FixedUpdateEarlyStopping`. Used to stop training automatically if progress plateaus.
   * - **Evaluation**
     - :class:`~noether.core.callbacks.BestMetricCallback`, :class:`~noether.core.callbacks.TrackOutputsCallback`. Specialized monitoring for tracked metrics.
When to Use What?
-----------------

1.  **Use existing callbacks** for standard tasks like logging, checkpointing, and validation. These are highly configurable via YAML.
2.  **Inherit from** :class:`~noether.core.callbacks.periodic.PeriodicCallback` if you need to perform an action at regular intervals (e.g., logging a custom internal state of the model).
3.  **Inherit from** :class:`~noether.core.callbacks.periodic.PeriodicDataIteratorCallback` if you need to run inference on a specific dataset and aggregate the results to compute a metric. Those callbacks need to configure a dataset key to specify which dataset to run on.
4.  **Inherit from** :class:`~noether.core.callbacks.base.CallbackBase` if your logic only needs to run once at the very beginning or end of training.

How to Configure Callbacks
--------------------------

Callbacks are usually defined in your experiment configuration under the ``callbacks`` key. Each callback requires the fully qualified class name  (e.g., ``noether.core.callbacks.progress.Progress``) as the ``kind``, exactly one frequency setting (``every_n_*``) and any additional parameters specific to that callback.

Example YAML configuration:

.. code-block:: yaml

   callbacks:
    - kind: noether.core.callbacks.CallbackClassName
      name: CallbackInstanceName
      every_n_epochs: 1
      # or every_n_updates: 1
      # additional_param: value


How to Implement Custom Callbacks
----------------------------------

To create a custom callback, define a new class that inherits from one of the base callback classes. Override the relevant methods to inject your logic at the desired points in the training process.


.. code-block:: python

    from noether.core.schemas.callbacks import PeriodicDataIteratorCallbackConfig 
    from noether.core.callbacks.periodic import PeriodicCallback
    from noether.data.samplers import SamplerIntervalConfig
    import torch

    class CustomCallbackConfig(PeriodicDataIteratorCallbackConfig):
        # Define any configuration parameters your callback needs
        

    class MyCustomCallback(PeriodicCallback):
        def __init__(self, callback_config: CustomCallbackConfig, **kwargs):
            super().__init__(callback_config, **kwargs)

        def register_sampler_config(self) -> SamplerIntervalConfig:
            # Define how to sample data for this callback. By default, this method takes 
            # the sampler config by using the dataset_key from the callback config.
            # If you need custom sampling logic, override this method.
            return SamplerIntervalConfig(...)

        def process_data(self, batch: dict[str, torch.Tensor], **_) -> dict[str, torch.Tensor]:
            model_output = self.model(**batch)
            # some more custom logic
            out = {"custom_output": model_output}
            return out

        def process_results(self, results: dict[str, torch.Tensor], **_) -> None:
            # this method gets the aggregated results of the process_data method across the dataset
            # do something with the results
            self.writer.add_scalar("custom_metric", results["custom_output"].mean().item())