Key Concepts
============

Config-driven vs Code-driven
----------------------------

The **Noether Framework** offers flexibility and lets you get started quickly with a config-driven workflow.
What does that mean?
In a config-driven workflow, you use text files (for example ``*.yaml``) to define how a training pipeline should run.
These configs are easy to store and reuse for experiment tracking: create a new folder and put your files there.

To avoid one big config file that becomes hard to manage, we suggest to split configs by module:

- ``callbacks``
- ``data_specs``
- ``dataset_normalizer``
- ``datasets``
- ``experiment``
- ``model``
- ``optimizer``
- ``pipeline``
- ``tracker``
- ``trainer``

You can see this structure in the ``tutorial`` folder of the ``noether`` repository. Each folder contains ``yaml``
files that define settings for datasets, models, trainers, and more.

When to use what
----------------

If you went through the ``tutorial`` and got your first training run working — nice! The next step is to adapt the
the examples of the **Noether Framework** to your needs. To do so you have three options:

- Go deeper with config changes (for example, change model size).
- Do basic code customization: create custom attention blocks, transformer blocks, etc., and use them from configs.
- Focus on code only: create a custom training step (a ``trainer``) for your domain.

Code customization is usually the most flexible option, but it can feel overwhelming if you are new to programming.
As a rule of thumb: start with configs and our pipelines, then move to deeper customizations when you need them.
