## `Noether` Starter Kit Project
----
This folder contains skeleton/boilerplate code for a minimal working `Noether` training pipeline, including all required components.

1.  A dataset that loads (and generates) dummy data.
2.  A pipeline that processes the data (i.e., sample processing, collating, batch processing) and collates it into a batch. sample processingand batch processing are skipped for this base project.
3.  A simple base model that is compatible with the `Noether` `BaseTrainer` class.
4.  A trainer that executes the forward pass of the model and computes a loss value.
5.  A callback that computes test statistics.
6.  Base configurations for all components.

You can take this codebase and start extending it or rebuilding components for your own project. This means: (1) adding custom data loading and collating, (2) building custom models, (3) implementing custom training loops, and (4) creating custom callbacks.

To run the training loop from the command line, simply execute the following command from the `/noether` root directory:

```bash
uv run noether-train --hp ./boilerplate_project/configs/base_experiment.yaml +seed=1 +devices=\"0\" tracker=disabled
```