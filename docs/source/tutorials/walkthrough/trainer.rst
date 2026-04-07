The Trainer
===========

The ``AerodynamicsCFDTrainer`` (in ``trainers/aerodynamics_cfd.py``)
is a specialized trainer designed for aerodynamics Computational Fluid Dynamics (CFD) tasks,
specifically for the AhmedML, DrivAerML, DrivAerNet++, ShapeNet-Car, and Emmi-Wing datasets.
Its primary role is to manage the training step by processing model outputs, computing a flexible
weighted loss, and returning the results.

For a step-by-step guide on implementing custom trainers, see
:doc:`/guides/training/implement_a_custom_trainer`.


BaseTrainer implementation
--------------------------

To implement a custom ``Trainer`` for a downstream project, you must extend the
:py:class:`~noether.training.trainers.BaseTrainer` class. The ``BaseTrainer`` handles the full training
loop and provides two key methods:

.. literalinclude:: ../../../../src/noether/training/trainers/base.py
   :language: python
   :pyobject: BaseTrainer.loss_compute
   :dedent:

.. literalinclude:: ../../../../src/noether/training/trainers/base.py
   :language: python
   :pyobject: BaseTrainer.train_step
   :dedent:

**Understanding the two key methods:**

As an end-user, you need to implement **loss_compute** and sometimes **train_step**.

The **train_step** method receives the batch from the multi-stage pipeline and the **model**
being trained (which can be a ``DistributedDataParallel`` model when training on multiple GPUs).

In the base implementation, the batch is split into two sub-batches:

#. **Forward batch**: Contains all tensors needed for the forward pass. The model receives the
   ``forward_batch`` as named keyword arguments, and the forward pass is computed.
#. **Targets batch**: Contains tensors needed for loss computation. The **loss_compute** method
   computes the custom loss for your task.

.. important::

   A warning is emitted if there are keys in the batch that do not end in either the forward
   batch or the target batch. This means that the collator returns tensors that are not used
   during the forward pass.

**Return value requirements:**

The **train_step** method must always return the :py:class:`~noether.training.trainers.TrainerResult` dataclass, which should
contain:

- A scalar value of the total loss used to compute gradients (can be a weighted sum of multiple losses)
- A dictionary with the losses you want to log
- Optionally, a dictionary with additional output for logging

**When to override train_step:**

The **train_step** method defined in the :py:class:`~noether.training.trainers.BaseTrainer`
class fits most general deep learning forward passes. However, you can decide whether this
implementation is sufficient for your downstream training task. If not, you can always implement
a custom **train_step** method in the child trainer class (as has been done in the scaffold
template at ``src/noether/scaffold/template_files/trainer/base.py``).


BaseTrainer configuration
-------------------------

When using the default **train_step** method, you must define both the **forward_properties**
and the **target_properties** to define which tensors are part of the ``forward_batch`` and
which tensors are part of the ``target_batch``.

In this walkthrough, the target properties are fixed per dataset, while the
**forward_properties** depend on the model. Therefore, we define them as follows:

**Full trainer configuration:**

The complete trainer config for ShapeNet-Car is defined in :source:`configs/trainer/shapenet_trainer.yaml <../../../../recipes/aero_cfd/configs/trainer/shapenet_trainer.yaml>`:

.. literalinclude:: ../../../../recipes/aero_cfd/configs/trainer/shapenet_trainer.yaml
   :language: yaml


AerodynamicsCFDTrainer implementation
-------------------------------------

The most important variables in the ``__init__`` method are the loss weights, which give you
fine-grained control over the training objective.

**Loss weight hierarchy:**

The loss has two levels of weights:

- **Individual weights**: Parameters like ``surface_pressure_weight`` and
  ``volume_velocity_weight`` control the importance of a specific physical quantity in the
  total loss.
- **Group weights**: The ``surface_weight`` and ``volume_weight`` parameters apply an additional
  weight to all surface-related or volume-related losses, respectively.

During initialization, the trainer uses these weights to build an internal ``loss_items`` list.
The ``output_modes`` parameter (e.g., ``['surface_pressure', 'volume_velocity']``) specifies
which of these potential losses should be computed during training.

**Custom loss calculation (loss_compute):**

This method contains the core logic of the trainer for computing the loss:

.. literalinclude:: ../../../../recipes/aero_cfd/trainers/aerodynamics_cfd.py
   :language: python
   :pyobject: AerodynamicsCFDTrainer.loss_compute
   :dedent:

It iterates through the ``loss_items`` configured during initialization. For each
item (like ``surface_pressure``), it checks that its weight is non-zero and that the model
produced a corresponding output key.

This flexible system allows you to easily experiment with different combinations of output
objectives without changing the underlying code.

When using only a single loss value, the ``loss_compute`` method is not needed and can be
implemented directly inside the forward function (by overriding the base ``train_step`` method,
as done in the
scaffold template at ``src/noether/scaffold/template_files/trainer/base.py``).
