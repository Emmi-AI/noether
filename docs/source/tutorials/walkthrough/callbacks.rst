Callbacks
=========

A **callback** is an object that can perform actions at various stages of the training loop,
such as at the beginning or end of training, an epoch, or an update step. Callbacks are the
most complex objects in the Noether Framework.

For a comprehensive guide on callback types and implementation, see
:doc:`/guides/training/use_callbacks`.


Overview
--------

The
``AeroMetricsCallback`` (in ``callbacks/aero_metrics.py``)
is a specific callback that runs the current model on a separate validation or test set,
computes error metrics, and logs them. This class inherits from
:py:class:`~noether.core.callbacks.periodic.PeriodicDataIteratorCallback`,
meaning its main logic is executed at regular intervals and iterates over a dataset.

In this walkthrough, we focus on
:py:class:`~noether.core.callbacks.periodic.PeriodicDataIteratorCallback`. However, you can
also implement a :py:class:`~noether.core.callbacks.periodic.PeriodicCallback`, which does not
iterate over a dataset but can be used, for example, to
store an exponential moving average (EMA) of the model weights.

**Callback access to training components:**

Callbacks have access to the following (among others):

- **The Trainer** (``self.trainer``): Provides access to trainer properties
- **The Model** (``self.model``): The currently trained model
- **The Data Container** (``self.data_container``): Object containing all datasets, allowing normalizers to be accessed for denormalization


Implementing PeriodicDataIteratorCallback
-----------------------------------------

Callbacks that inherit from
:py:class:`~noether.core.callbacks.periodic.PeriodicDataIteratorCallback` must implement two
methods:

#. ``process_data(self, batch: dict[str, torch.Tensor], **_) -> dict[str, torch.Tensor]``:
   Receives a batch from the dataset as input and computes metrics (or tensors) that are returned.
#. ``process_results(self, results: dict[str, torch.Tensor], **_) -> None``:
   All computed metrics/tensors from the ``process_data`` method are aggregated into a dictionary
   and processed by this method.

For example, the ``process_results`` method can use ``self.writer`` to log metrics to
Weights & Biases.

**process_data implementation:**

The ``process_data`` method of the ``AeroMetricsCallback`` simply looks like:

.. literalinclude:: ../../../../recipes/aero_cfd/callbacks/aero_metrics.py
   :language: python
   :pyobject: AeroMetricsCallback.process_data
   :dedent:

First, it computes the model outputs, and next, it adds the desired metrics to an output
dictionary. All the substeps are implemented by individual methods in the callback itself.
See the full implementation in :source:`callbacks/aero_metrics.py <../../../../recipes/aero_cfd/callbacks/aero_metrics.py>` for details.


Configuring callbacks
---------------------

In :source:`configs/trainer/shapenet_trainer.yaml <../../../../recipes/aero_cfd/configs/trainer/shapenet_trainer.yaml>`, we define the list of callbacks to use for the
trainer class (for ShapeNet-Car). Below are three callback configurations:

.. literalinclude:: ../../../../recipes/aero_cfd/configs/callbacks/training_callbacks_shapenet.yaml
   :language: yaml

**Periodic callback triggers:**

To define how often a periodic callback should be triggered, set one of the following arguments
in your configuration:

- ``every_n_epochs``: Triggers the callback every N epochs
- ``every_n_updates``: Triggers the callback every N model update steps
- ``every_n_samples``: Triggers the callback after every N samples have been processed

You cannot define multiple of these arguments. In addition to the interval, you can also define
the ``batch_size``, which is usually set to ``1`` to compute metrics per sample.

**Required callback parameters:**

For all periodic callbacks, you must define:

- ``dataset_key``: Indicates which dataset (configured earlier) should be used to run the callback
- ``name``: Must match a name in the callback schemas so that the correct schema can be used for data validation


Denormalization for metrics
---------------------------

Metrics are usually computed on unnormalized data. To denormalize the normalization steps
executed by the dataset, we retrieve the data normalizers via the ``DataContainer``. In the
``__init__`` method of the callback we implement, we use the available ``self.data_container``
to get the correct dataset used for this callback and retrieve the normalizers:

.. literalinclude:: ../../../../recipes/aero_cfd/callbacks/aero_metrics.py
   :language: python
   :lines: 89-95
   :dedent:

To denormalize predictions, the ``_denormalize`` method looks up the normalizer by key and
calls ``inverse``:

.. literalinclude:: ../../../../recipes/aero_cfd/callbacks/aero_metrics.py
   :language: python
   :pyobject: AeroMetricsCallback._denormalize
   :dedent:


Computed metrics
----------------

For each output in the ``AeroMetricsCallback``, we calculate the following metrics:

#. **Mean Squared Error (MSE)**: The average of the squared differences between the prediction
   and the target.
#. **Mean Absolute Error (MAE)**: The average of the absolute differences between the prediction
   and the target.
#. **Relative L2 Error**: The Euclidean norm of the error vector divided by the norm of the
   target vector, measuring the error relative to the magnitude of the ground truth.


Final evaluation with repeated testing
---------------------------------------

At the end of training, we want to run the model one more time on the test set, looping 10
times over that set to reduce variance due to the point sampling. Earlier, we configured the
``test_repeat`` dataset in ``shapenet_dataset.yaml``, which uses the ``RepeatWrapper`` to loop
over the dataset with 10 repetitions. We can now use ``test_repeat`` with this custom dataset
implementation for the final callback.

Moreover, we set ``every_n_epochs: ${trainer.max_epochs}`` to ensure that this callback is only
executed at the very end. Each metric is logged with the corresponding ``dataset_key`` to
Weights & Biases.

For the ``CAEML`` dataset, we also implemented chunked inference, where we loop over the entire
surface and volume mesh in chunks to do inference on the full mesh. To enable this, we set
``chunked_inference: true``, and we configured a dataset ``chunked_test`` which has a
multi-stage pipeline that returns all points in the surface/volume mesh.
