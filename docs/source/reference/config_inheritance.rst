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