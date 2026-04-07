Models
======

Building models in the Noether Framework is straightforward and follows the same patterns as
standard PyTorch models that inherit from ``torch.nn.Module``.

To be compatible with the Noether Trainer, all models must inherit from
:py:class:`~noether.core.models.Model` (or :py:class:`~noether.core.models.CompositeModel` for
multi-component architectures, discussed later). Beyond this, a model is implemented just like any PyTorch
model: define layers in the constructor (``__init__``) and implement the ``forward`` method.

For a step-by-step guide on implementing custom models, see
:doc:`/guides/training/implement_a_custom_model`.


The ``ModelBaseConfig`` schema
------------------------------

Each model in the Noether Framework must inherit from the :py:class:`~noether.core.models.Model`
class. The config schema for models is defined by
:py:class:`~noether.core.schemas.models.base.ModelBaseConfig`:

.. literalinclude:: ../../../../src/noether/core/schemas/models/base.py
   :language: python
   :pyobject: ModelBaseConfig
   :end-before: @property
   :dedent:

**Key configuration parameters:**

- ``kind``: The full class path to the model class (e.g., ``noether.modeling.models.AeroTransformer``).
- ``name``: A unique identifier for the model, typically overridden in child config classes to match the correct model configuration.
- ``optimizer_config``: The optimizer configuration for training. Can be ``None`` when loading a model for inference only.
- ``initializers``: Optional list of initializer configs for loading pre-trained weights or custom weight initialization.
- ``is_frozen``: Boolean flag to freeze all model parameters (useful for transfer learning or ensemble models).
- ``forward_properties``: List of properties to be used as inputs for the model's forward pass. Only relevant when using the ``BaseTrainer``'s default ``train_step`` method.

.. note::

   In the Noether Framework, optimizers are attached to models rather than being global. This
   design allows different components of composite models to use different optimizers and
   learning rates.


Implementing a custom model
----------------------------

A minimal custom model implementation looks as follows:

**Python implementation:**

.. code-block:: python

   from noether.core.models import Model


   class CustomModel(Model):
       def __init__(self, model_config: CustomModelConfig, **kwargs):
           # the model config needs to be passed to the parent Model class
           super().__init__(model_config=model_config, **kwargs)

           self.config = model_config

           # Define your model layers here
           self.encoder = torch.nn.Linear(
               model_config.input_dim, model_config.hidden_dim
           )
           self.decoder = torch.nn.Linear(
               model_config.hidden_dim, model_config.output_dim
           )

       def forward(
           self, input_tensor: torch.Tensor
       ) -> dict[str, torch.Tensor]:
           """
           Forward pass of the model.

           Args:
               input_tensor: torch tensor with data

           Returns:
               Dictionary containing model outputs.
           """
           x = input_tensor

           # Forward pass
           hidden = self.encoder(x)
           output = self.decoder(torch.nn.functional.relu(hidden))

           return {"output": output}

**Corresponding YAML configuration:**

.. code-block:: yaml

   kind: path.to.CustomModel
   name: custom_model
   input_dim: 3
   hidden_dim: 128
   output_dim: 1
   optimizer_config: ${optimizer}
   forward_properties:
     - input_tensor


Aerodynamic model wrappers
--------------------------

The Noether Framework provides ready-to-use aerodynamic model wrappers in
:py:mod:`noether.modeling.models.aerodynamics`. These wrappers (``AeroTransformer``,
``AeroTransolver``, ``AeroUPT``, ``AeroABUPT``) inherit from :py:class:`~noether.core.models.Model`
and add common utilities for CFD tasks:

#. **Surface and volume bias projection**: An MLP projection layer to handle domain-specific biases.
#. **Physics feature projection**: A linear layer to map physics features (e.g., SDF, normals) to the model's hidden dimension.
#. **Positional embeddings**: Sine-cosine or linear positional embedding layers for input coordinates.
#. **Output projection**: A final linear layer to project from the hidden dimension to the number of predicted physical quantities.


Handling model outputs
----------------------

Physical quantities predicted for surface points often differ from those for volume points.
For example:

- **Surface predictions**: pressure, wall shear stress
- **Volume predictions**: velocity, pressure, vorticity

The ``_gather_outputs`` function in ``noether.modeling.models.aerodynamics`` handles this
heterogeneity:

- Takes the entire output tensor and the number of surface points
- Splits the output tensor to isolate surface predictions from volume predictions
- Uses ``AeroDataSpecs`` to map output dimensions to named physical quantities

**Example output structure:**

.. code-block:: python

   {
       "surface_pressure": tensor[...],   # dimension 0 of surface outputs
       "volume_velocity": tensor[...],    # dimensions 1:4 of volume outputs
       "surface_friction": tensor[...],   # dimension 4:7 of surface outputs
   }

By using ``_gather_outputs`` consistently across all models, the output dictionary is structured
in a way that the trainer's ``loss_compute`` method can process uniformly. This design allows
the same trainer to work with all model architectures without modification.


Composite Models
----------------

A **composite model** consists of multiple :py:class:`~noether.core.models.Model` sub-modules, each
potentially with its own:

- Optimizer and learning rate / learning rate schedule
- Weight initialization strategy
- Frozen/trainable status

**Example:** The
``CompositeTransformer`` (in ``model/composite_transformer.py``)
demonstrates a Transformer model with two sub-modules, each with independent configurations.

**Configuration example:**

.. literalinclude:: ../../../../recipes/aero_cfd/configs/model/composite_transformer.yaml
   :language: yaml


Noether model zoo
-----------------

The Noether Framework includes base implementations for several state-of-the-art models.
For a complete listing, see :doc:`/noether/model_zoo`.

.. list-table::
   :widths: 15 20 30 35
   :header-rows: 1

   * - Model
     - Paper
     - Implementation
     - Notes
   * - **AB-UPT**
     - `arXiv:2502.09692 <https://arxiv.org/abs/2502.09692>`_
     - ``AeroABUPT``
     - Aerodynamic wrapper around :py:class:`~noether.modeling.models.ab_upt.AnchoredBranchedUPT`
   * - **Transformer**
     - ---
     - ``AeroTransformer``
     - Aerodynamic wrapper with RoPE support
   * - **Transolver**
     - `arXiv:2402.02366 <https://arxiv.org/abs/2402.02366>`_
     - ``AeroTransolver``
     - Aerodynamic wrapper around :py:class:`~noether.modeling.models.transolver.Transolver`
   * - **Transolver++**
     - `arXiv:2502.02414 <https://arxiv.org/abs/2502.02414>`_
     - Schema only
     - Extension of Transolver with different attention class
   * - **UPT**
     - `arXiv:2402.12365 <https://arxiv.org/abs/2402.12365>`_
     - ``AeroUPT``
     - Aerodynamic wrapper around :py:class:`~noether.modeling.models.upt.UPT`

All aerodynamic model wrappers are implemented in
:py:mod:`noether.modeling.models.aerodynamics`. They wrap the base model implementations and
add the aerodynamic-specific input/output handling (positional embeddings, physics features,
output gathering).
