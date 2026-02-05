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