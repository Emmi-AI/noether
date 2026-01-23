Design Principles and Limitations
=================================

This document outlines the core design principles, architectural patterns, and known limitations of the Noether
Framework. It serves as a guide for developers to understand the "why" behind the code structure.

Design Principles
-----------------

The framework is built upon five major pillars aimed at robustness, reproducibility, and clear separation of concerns.

1. Configuration-Driven Development (CDD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle:** Behavior is strictly separated from implementation.

Almost every component in the system (Datasets, Models, Layers) is instantiated via a specific **Configuration Object**
rather than raw arguments.

**Implementation:**
We rely heavily on Pydantic models (e.g., ``DatasetBaseConfig``, ``LinearProjectionConfig``) to define the expected
inputs. Components are rarely instantiated directly; instead, they are built by passing a config object
to a constructor or factory.

.. code-block:: python

    # Example: A TransformerBlock receives a full config object, not individual args.
    block = TransformerBlock(config=my_block_config)

2. Factory Pattern & Dynamic Instantiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle:** Decoupling configuration from import logic.

The framework uses a centralized "Factory" mechanism to build objects. It relies on **Class Paths** (strings)
to resolve classes at runtime, avoiding heavy import chains at the top level.

**Implementation:**
The ``Factory.instantiate`` method uses the ``kind`` field in configurations to locate the correct class constructor.

.. code-block:: python

   # Get kind from the config:
   kind = object_config.kind

   # Factory logic: get the class contructor and pass the config to the class when instantiating 
   class_constructor = class_constructor_from_class_path(kind)
   instance = class_constructor(object_config, **kwargs)

3. Strict Type Safety & Runtime Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle:** Catch errors at initialization, not during training.

We use Python type hints (``list[torch.Tensor]``, ``Literal``) and Pydantic validators to ensure invalid
states are impossible to represent.

**Implementation:**

- **Pydantic Validators:** ``@model_validator(mode="after")`` ensures fields like, for example, ``ndim`` are valid (e.g., only 1, 2,
or 3).
- **Type Guards:** Utility functions like ``validate_path`` allow strictly typed filesystem operations.

4. Defensive Programming
~~~~~~~~~~~~~~~~~~~~~~~~

**Principle:** Do not trust the input.

The code frequently asserts state validity and checks types explicitly at runtime, even when type hints are present,
to prevent silent failures in complex pipelines.

**Implementation:**

- Explicit ``isinstance`` checks in factories.
- State guards (e.g., ``assert self._stats is not None``) in stateful classes like ``RunningMoments``.

5. Composition over Inheritance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle:** Complex behaviors are built by wrapping simple objects.
We avoid deep inheritance trees. Instead, we use wrappers and composition to add functionality like caching,
normalization, or logging.

**Implementation:**
The ``DatasetFactory`` does not subclass datasets to add features; it wraps a base dataset with a list of
``dataset_wrappers`` (e.g., normalizers).

Limitations & Trade-offs
~~~~~~~~~~~~~~~~~~~~~~~~

While the architecture ensures robustness, it introduces specific trade-offs that developers must be aware of.

- **Tight Coupling of Configs to Code:**
    Many classes accept a single ``config`` object in their ``__init__``. This makes it difficult to use these modules
    "standalone" (e.g., in a notebook) without first constructing the specific Pydantic configuration object they
    expect.

- **Circular Dependency Risks:**
    Configuration schemas need to know about classes (for validation), and classes need to know about schemas
    (for typing). This occasionally forces the use of string-based class resolution instead of direct imports to avoid
    cycles.

- **"Stringly" Typed Architecture:**
    The reliance on string paths (e.g., ``kind="torch.optim.SGD"``) for instantiation means that automated
    refactoring tools (like "Rename Class" in IDEs) may miss references in YAML or Config files.

- **Factory Indirection:**
    The ``Factory.instantiate`` method contains implicit logic (e.g., checking kwargs for "kind" if the config is None).
    This "magic" can sometimes obscure how exactly an object is being created during debugging.

- **Boilerplate Overhead:**
    Adding a simple new feature often requires modifying three distinct files: the **Implementation Class**,
    the **Configuration Schema**, and the **Factory** logic. This favors stability over rapid prototyping speed.

- **Nested Configuration Schema complexity:**
    In our root config schule, we define the modules needed for the `Noether Framework`, each module having their own config schema. 
    This can lead to deeply nested configurations that can be hard to navigate and understand in the beginning, especially for new users.
