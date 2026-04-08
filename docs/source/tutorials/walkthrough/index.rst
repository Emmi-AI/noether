Walkthrough: Aerodynamic CFD with Noether
==========================================

**Authors**: Maurits Bleeker, Markus Hennerbichler, and Pavel Kuksa

This walkthrough demonstrates how to use the Noether Framework through a practical project
based on the experiments from Section 4.4 of the `AB-UPT paper <https://arxiv.org/pdf/2502.09692>`_.
While this walkthrough covers the core functionality of the framework, it does not cover every
possible feature or use case.

.. note::

   This walkthrough presents recommended practices and patterns, but it is not a full blueprint
   on how to use the framework. The Noether Framework is flexible and supports multiple approaches
   to implementing the same functionality.

The full source code for this walkthrough lives in the ``recipes/aero_cfd/``
directory of the repository.

.. toctree::
   :maxdepth: 2

   introduction
   configuration
   dataset
   pipeline
   trainer
   models
   callbacks
   running_experiments
