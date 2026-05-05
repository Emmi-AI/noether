Configuring AB-UPT
==================

This guide walks through how to configure
:py:class:`~noether.modeling.models.ab_upt.AnchoredBranchedUPT` via
:py:class:`~noether.core.schemas.models.AnchorBranchedUPTConfig`. It is structured around
the three stages of the model — the geometry encoder, the physics trunk, and the
per-domain decoder — and references the YAML configs shipped with the ``aero_cfd`` and
``heat_transfer`` recipes for concrete examples.

For background on the model itself, see the
`AB-UPT paper <https://arxiv.org/abs/2502.09692>`_ and :doc:`/noether/model_zoo`.


At a glance
-----------

A minimal AB-UPT YAML config looks like this (from
:source:`recipes/heat_transfer/configs/model/ab_upt.yaml <../../../../recipes/heat_transfer/configs/model/ab_upt.yaml>`):

.. literalinclude:: ../../../../recipes/heat_transfer/configs/model/ab_upt.yaml
   :language: yaml

The key parameters are:

- ``hidden_dim`` — shared across the geometry encoder, physics trunk, and decoder.
  Auto-injected into ``transformer_block_config`` and ``supernode_pooling_config`` via
  :doc:`/reference/config_inheritance`, so you only set it once.
- ``transformer_block_config`` — attention head count, MLP expansion, RoPE settings.
  AB-UPT requires ``use_rope: true``.
- ``supernode_pooling_config`` — geometry encoder front-end. Only required if
  ``physics_blocks`` contains a ``perceiver`` block.
- ``physics_blocks`` — ordered list of block types (see below).
- ``num_domain_decoder_blocks`` — per-domain self-attention head depth.
- ``data_specs`` — defines the domains and their input/output fields. Drives the
  per-domain decoder projections automatically via the ``domain_decoder_configs``
  computed field.


Stage 1: the geometry encoder
-----------------------------

The geometry encoder is **only instantiated when the physics trunk contains a
``perceiver`` block** and ``supernode_pooling_config`` is set. It runs a
:py:class:`~noether.modeling.modules.encoders.SupernodePooling` encoder over the
geometry mesh, followed by ``geometry_depth`` standard transformer blocks.

In the heat-transfer config above, ``geometry_depth: 1`` and a single ``perceiver``
block in ``physics_blocks`` is enough to attend to the geometry encoding once at the
start of the trunk.

Skip the geometry branch entirely by:

- Removing all ``perceiver`` / ``perceiver_untied`` entries from ``physics_blocks``, **or**
- Leaving ``supernode_pooling_config`` unset.

In that case, the model operates purely on per-domain anchor positions.


Stage 2: the physics trunk (``physics_blocks``)
-----------------------------------------------

``physics_blocks`` is the ordered list of attention blocks applied to the concatenated
per-domain anchor tokens. Each entry is one of:

.. list-table::
   :widths: 18 12 70
   :header-rows: 1

   * - Block
     - Weights
     - Description
   * - ``self``
     - shared
     - Self-attention **within each domain branch** independently. Token mixing does
       not cross domain boundaries.
   * - ``cross``
     - shared
     - Cross-attention from each domain to all **other** domains' anchors. Lets
       domains exchange information.
   * - ``joint``
     - shared
     - Full self-attention over **all** anchors from **all** domains jointly.
   * - ``perceiver``
     - shared
     - Cross-attention from anchors to the geometry encoding. Requires the geometry
       branch.
   * - ``self_untied`` / ``cross_untied`` / ``joint_untied`` / ``perceiver_untied``
     - **per-domain**
     - Same attention pattern as the un-suffixed variant, but with **separate weights
       for each domain** (wrapped in ``UntiedTransformerBlock`` /
       ``UntiedPerceiverBlock``).

.. note::

   ``shared`` is a deprecated alias for ``self``; it still works but emits a
   ``DeprecationWarning`` and will be removed in a future release.

Choosing tied vs. untied
~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``_untied`` variants when domains have **substantially different statistics**
(e.g. surface vs. volume in CFD) and you have enough data to fit per-domain weights.
Tied (shared) blocks are smaller and regularize across domains; untied blocks have
more capacity per domain at the cost of parameters.

Concrete patterns
~~~~~~~~~~~~~~~~~

**Aerodynamics (two domains, surface + volume)** — from
:source:`recipes/aero_cfd/configs/model/ab_upt.yaml <../../../../recipes/aero_cfd/configs/model/ab_upt.yaml>`:

.. literalinclude:: ../../../../recipes/aero_cfd/configs/model/ab_upt.yaml
   :language: yaml
   :lines: 12-22

A single ``perceiver`` block reads geometry once, then the trunk alternates
``self`` (in-domain mixing) with ``cross`` (cross-domain exchange) for four cycles,
ending in a ``self`` block.

**Heat transfer (single volume domain)** — from
:source:`recipes/heat_transfer/configs/model/ab_upt.yaml <../../../../recipes/heat_transfer/configs/model/ab_upt.yaml>`:

.. literalinclude:: ../../../../recipes/heat_transfer/configs/model/ab_upt.yaml
   :language: yaml
   :lines: 12-18

With only one domain, ``cross`` and ``joint`` are equivalent to ``self``, so the
trunk is a stack of ``perceiver`` + ``self`` blocks.


Stage 3: optional per-domain decoder
------------------------------------

After the physics trunk, ``num_domain_decoder_blocks`` optionally adds **untied self-attention
blocks per domain** (no weight sharing across domains), then a linear projection to
that domain's output fields. This is semantically equivalent to adding the same number of
``self_untied`` blocks to the end of the trunk for each domain. The only difference is,
that the decoder block depth can be set per-domain.
The output fields and their slicing are derived from
``data_specs.domains[name].output_dims``.

.. code-block:: yaml

   num_domain_decoder_blocks:
     surface: 2
     volume: 2

Set per-domain depths to ``0`` (or omit the entry) to skip the decoder block stack
and project directly from the trunk output. The output projection itself is always
present.


Other knobs
-----------

- ``init_weights`` — weight initialization mode for linear layers; defaults to
  ``"truncnormal002"``.
- ``drop_path_rate`` — stochastic depth rate; defaults to ``0.0``.
- ``data_specs.conditioning_dims`` — when set, the total conditioning dimension is
  pushed into ``transformer_block_config.condition_dim`` and used by the
  conditioning path of every transformer/perceiver block. The heat-transfer recipe
  uses this to condition on simulation parameters.


Putting it all together
-----------------------

End-to-end training configs (model + dataset + pipeline + trainer + callbacks) for
both recipes:

- Aerodynamics: `recipes/aero_cfd/configs/
  <https://github.com/Emmi-AI/noether/tree/main/recipes/aero_cfd/configs>`_ —
  see ``train_*.yaml`` for per-dataset entry points and
  ``experiment/<dataset>/ab_upt.yaml`` for AB-UPT-specific overrides.
- Heat transfer: `recipes/heat_transfer/configs/
  <https://github.com/Emmi-AI/noether/tree/main/recipes/heat_transfer/configs>`_ —
  start from ``train_simshift_heatsink.yaml`` and the
  ``+experiment/simshift_heatsink=ab_upt`` override.

Both recipes are wired up to the ``noether-train`` CLI; see
:doc:`/guides/working_with_cli` and :doc:`/guides/training/launch_job` for how to run
them locally or on SLURM.
