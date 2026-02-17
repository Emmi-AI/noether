Experiment Tracking
===================

Experiment tracking is a crucial part of the machine learning workflow. It allows you to log metrics over the course of training, making it easier to compare different runs and reproduce results.

The Noether Framework supports multiple experiment trackers through a unified interface.

By configuring a tracker for your run, all callbacks that use the tracker will automatically log metrics to the selected tracking service.

Supported Trackers
------------------

Currently, Noether supports the following trackers:

*   **Weights & Biases (W&B)**: A popular developer-first MLOps platform.
*   **Trackio**: A HuggingFace-integrated tracker for lightweight experiment logging.
*   **TensorBoard**: A classic, local-first visualization toolkit for machine learning experiments.

Configuring Trackers
--------------------

Trackers are configured in the `tracker` section of your experiment configuration.

Weights & Biases
~~~~~~~~~~~~~~~~

To use W&B, use the ``noether.core.trackers.WandBTracker`` kind:

.. code-block:: yaml

    kind: noether.core.trackers.WandBTracker
    project: my-project
    entity: my-team
    mode: online  # Can be 'online', 'offline', or 'disabled'

Trackio
~~~~~~~

:: note ::
    You have to install the `trackio` package separately to use this tracker: ``pip install trackio``.

To use `Trackio <https://huggingface.co/docs/trackio/en/index>`__, use the ``noether.core.trackers.TrackioTracker`` kind:

.. code-block:: yaml

    kind: noether.core.trackers.TrackioTracker
    project: my-project
    space_id: my-hf-space-id  # Optional: defaults to your HF space if running on HF

TensorBoard
~~~~~~~~~~~

:: note ::
    You have to install the `tensorboard` package separately to use this tracker: ``uv pip install tensorboard``. 

To use TensorBoard, use the ``noether.core.trackers.TensorboardTracker`` kind:

.. code-block:: yaml

    kind: noether.core.trackers.TensorboardTracker
    log_dir: tensorboard_logs         # Optional: defaults to tensorboard_logs. Directory to store TensorBoard event files. This directory will be created inside output_path.
    flush_secs: 60                     # Optional: defaults to 60 seconds. Specifies how often to flush pending events to disk


Start the TensorBoard server from your terminal and point to the output directory specified in your configuration:

.. code-block:: bash

    tensorboard --logdir ./outputs # Adjust the path to match your output directory

Once the server starts running, it will output a local URL, which you can open in your web browser to view your experiment dashboards.

Disabling Tracker
~~~~~~~~~~~~~~~~

To disable tracking, you can just set tracker to `None` or `null` in YAML:

Using Trackers in Callbacks
----------------------


All trackers inherit from :class:`~noether.core.trackers.base.BaseTracker` and provide a simple API for logging:

.. code-block:: python

    tracker.log({"loss": 0.5, "accuracy": 0.9})

Trackers are automatically initialized by the trainer and made available to all callbacks. 
You can access the tracker in your custom callbacks with the :attr:`~noether.core.callbacks.base.CallbackBase.tracker` attribute (`self.tracker`).