Configuration Inheritance
=========================

UPT and AB-UPT models support automatic configuration injection from parent to submodules for shared parameters. This feature helps reduce verbosity and maintain consistency across complex model architectures.

How It Works
------------

When you define certain parameters at the top level of a configuration, they automatically propagate down to nested submodules unless explicitly overridden in the submodule configuration. This inheritance works recursively across multiple levels of nesting.

**Inherited parameters:**

- **UPT models**: ``hidden_dim``, ``num_heads``, ``mlp_expansion_factor``
- **AB-UPT models**: ``hidden_dim``

The inheritance mechanism uses Pydantic's validation system and the ``InjectSharedFieldFromParentMixin`` to detect and propagate these shared fields before model instantiation.

Benefits
--------

- **Reduced redundancy**: Define shared parameters once instead of repeating them in every submodule
- **Consistency**: Ensures all submodules use the same architectural parameters by default
- **Flexibility**: Override inherited values in specific submodules when needed

Example - UPT Configuration
---------------------------

.. code-block:: yaml

   kind: tutorial.model.UPT
   name: upt
   hidden_dim: 192 
   num_heads: 3 
   mlp_expansion_factor: 4  
   approximator_depth: 12
   use_rope: true

   supernode_pooling_config:
     input_dim: 3
     radius: 9
     # hidden_dim: 192 (inherited from parent)
     # num_heads: 3 (inherited from parent)
     # mlp_expansion_factor: 4 (inherited from parent)

   approximator_config:
     use_rope: true
     # hidden_dim: 192 (inherited from parent)
     # num_heads: 3 (inherited from parent)
     # mlp_expansion_factor: 4 (inherited from parent)

   decoder_config:
     depth: 12
     input_dim: 3
     perceiver_block_config:
       use_rope: true
       # hidden_dim: 192 (inherited from parent via decoder_config)
       # num_heads: 3 (inherited from parent via decoder_config)
       # mlp_expansion_factor: 4 (inherited from parent via decoder_config)

Nested Inheritance
------------------

Configuration inheritance works across multiple levels. In the example above, ``perceiver_block_config`` is nested inside ``decoder_config``, which is nested in the top-level UPT config. The shared parameters propagate all the way down:

.. code-block:: text

   UPT config (hidden_dim=192)
     └── decoder_config (inherits hidden_dim=192)
           └── perceiver_block_config (inherits hidden_dim=192)

Overriding Inherited Values
---------------------------

You can override inherited values at any level by explicitly specifying them:

.. code-block:: yaml

   kind: tutorial.model.UPT
   name: upt
   hidden_dim: 192
   num_heads: 3
   mlp_expansion_factor: 4

   approximator_config:
     # hidden_dim: 192 (still inherited)
     # num_heads: 3 (still inherited)
     mlp_expansion_factor: 2 # Override: use 2 instead of inherited 4

When Inheritance Doesn't Apply
------------------------------

Configuration inheritance only works for:

- Parameters that are defined as "shared" in the model schema
- Submodules that have matching parameter names in their schema
- Dictionary-based configurations (if you define config with python code where submodules are instantiated with explicit parameters, inheritance won't apply)


If a submodule doesn't have a field matching the parent's shared parameter, that parameter simply isn't injected.

How to Add It to Your Own Schemas
----------------------------------

To add configuration inheritance to your own schemas, follow these steps:

1. **Add the mixin to your parent config**

   Import and inherit from ``InjectSharedFieldFromParentMixin`` in your parent configuration class:

   .. code-block:: python

      from pydantic import BaseModel, Field
      from noether.core.schemas.mixins import InjectSharedFieldFromParentMixin, Shared

      class MyModelConfig(BaseModel, InjectSharedFieldFromParentMixin):
          hidden_dim: int = Field(..., ge=1)
          num_layers: int = Field(..., ge=1)
          # ...other fields

2. **Mark sub-config fields with the Shared annotation**

   Use ``Annotated[SubConfigType, Shared]`` to mark which sub-config fields should receive inherited parameters:

   .. code-block:: python

      from typing import Annotated

      class MyModelConfig(BaseModel, InjectSharedFieldFromParentMixin):
          hidden_dim: int = Field(..., ge=1)
          num_layers: int = Field(..., ge=1)
          
          # This sub-config will receive inherited fields
          encoder_config: Annotated[EncoderConfig, Shared]
          
          # This sub-config will also receive inherited fields
          decoder_config: Annotated[DecoderConfig, Shared]

3. **Ensure sub-configs have matching field names**

   Only fields with matching names will be inherited. If your sub-config has a ``hidden_dim`` field and the parent has a ``hidden_dim`` field, the value will be inherited:

   .. code-block:: python

      class EncoderConfig(BaseModel):
          hidden_dim: int = Field(..., ge=1)  # Will inherit from parent
          depth: int = Field(..., ge=1)  # Won't inherit (no matching parent field)

4. **For nested inheritance, add the mixin to sub-configs too**

   If your sub-config also has nested configurations, add the mixin to enable multi-level inheritance:

   .. code-block:: python

      class DecoderConfig(InjectSharedFieldFromParentMixin, BaseModel):
          hidden_dim: int = Field(..., ge=1)
          
          # This nested config will also receive inherited fields
          attention_config: Annotated[AttentionConfig, Shared]

Complete Example
^^^^^^^^^^^^^^^^

.. code-block:: python

   from typing import Annotated
   from pydantic import BaseModel, Field
   from noether.core.schemas.mixins import InjectSharedFieldFromParentMixin, Shared

   class AttentionConfig(BaseModel):
       hidden_dim: int = Field(..., ge=1)
       num_heads: int = Field(..., ge=1)

   class EncoderConfig(InjectSharedFieldFromParentMixin, BaseModel):
       hidden_dim: int = Field(..., ge=1)
       depth: int = Field(..., ge=1)
       attention_config: Annotated[AttentionConfig, Shared]

   class MyModelConfig(BaseModel, InjectSharedFieldFromParentMixin):
       hidden_dim: int = Field(256, ge=1)
       num_heads: int = Field(8, ge=1)
       encoder_config: Annotated[EncoderConfig, Shared]

With this setup, a YAML configuration like:

.. code-block:: yaml

   hidden_dim: 256
   num_heads: 8
   encoder_config:
     depth: 6
     attention_config:
       # hidden_dim and num_heads inherited from top level

will automatically propagate ``hidden_dim`` and ``num_heads`` to both ``encoder_config`` and ``encoder_config.attention_config``.