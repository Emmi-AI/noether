.. meta::
   :description: Noether is a PyTorch-based framework for Engineering AI: models, datasets, training recipes, and tooling for physics-based ML research.
   :keywords: Noether, Emmi AI, Engineering AI, PyTorch, machine learning, scientific ML, neural operators, physics-based ML, surrogate models, simulation, CFD, CAE

Noether Framework Documentation
===============================

Welcome to the Noether Framework documentation. Here you will find available APIs, CLIs, etc.

.. grid:: 1
   :gutter: 2
   :class-container: sd-mb-4

   .. grid-item-card:: :ph:`rocket-launch` Start Here: Introduction to Noether
      :link: noether/introduction_to_noether_framework
      :link-type: doc
      :class-card: sd-bg-primary sd-text-white sd-shadow-lg

      A high-level overview of the core architecture, key concepts, and design principles.

.. toctree::
   :maxdepth: 2
   :caption: The Noether Framework
   :hidden:

   noether/introduction_to_noether_framework
   noether/design_principles_and_limitations
   noether/key_concepts
   noether/model_zoo
   noether/dataset_zoo
   noether/recipe_zoo
   noether/understanding_the_data_pipeline


.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/getting_started_install_and_verify
   tutorials/training_first_model_with_configs
   tutorials/training_first_model_with_code

   tutorials/walkthrough/index
   tutorials/scaffolding_a_new_project


.. toctree::
   :maxdepth: 2
   :hidden:

   How-to Guides <guides/index>

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   tutorials/prerequisites
   noether/io/caching
   reference/hardware_setup
   reference/config_inheritance
   API Reference <../autoapi/noether/index>


.. grid:: 1 2 2 2
   :gutter: 2
   :class-container: sd-equal-height

   .. grid-item::
      .. card:: :ph:`cube` The Noether Framework
        :class-card: sd-h-100
        :link: noether/index
        :link-type: doc
        :shadow: md

        Go deep directly. Learn about our core architecture, key concepts, and the design principles behind the framework.

   .. grid-item::
      .. card:: :ph:`graduation-cap` Tutorials
        :class-card: sd-h-100
        :link: tutorials/index
        :link-type: doc
        :shadow: md

        Guided lessons to install the framework, run your first simulation, and understand the basics.

   .. grid-item::
      .. card:: :ph:`toolbox` How-to Guides
        :class-card: sd-h-100
        :link: guides/index
        :link-type: doc
        :shadow: md

        Step-by-step recipes for common problems, like loading custom data, using private sources, or writing your
        own data collators.

   .. grid-item::
      .. card:: :ph:`book-open-text` Reference
        :class-card: sd-h-100
        :link: ../autoapi/noether/index
        :link-type: doc
        :shadow: md

        Technical lookup for complete API documentation


Recipes
-------

End-to-end training and evaluation examples for specific engineering tasks.

.. grid:: 1
   :gutter: 3
   :class-container: sd-mb-3

   .. grid-item-card:: :ph:`car-profile` AB-UPT Showcase on DrivAerML
      :link: tutorials/walkthrough/showcase
      :link-type: doc
      :class-card: sd-shadow-lg sd-rounded-3
      :class-title: sd-fs-4 sd-fw-bold sd-text-primary
      :class-body: sd-p-4

      :bdg-success:`Featured`

      Train AB-UPT on the DrivAerML dataset with a preset-based CLI.
      Ready-to-use model sizes for CPU, GPU, and Apple Silicon.

      +++

      **Open the showcase walkthrough →**

.. grid:: 1 2 2 2
   :gutter: 2
   :class-container: sd-equal-height

   .. grid-item-card:: :ph:`wind` External Aerodynamics
      :link: noether/aero_cfd_python
      :link-type: doc
      :class-card: sd-h-100 sd-shadow-sm sd-rounded-3
      :class-title: sd-fs-5 sd-fw-bold sd-text-primary
      :class-body: sd-p-4

      Multi-dataset aero CFD scripts (ShapeNet-Car, AhmedML, DrivAerML, DrivAerNet++, Emmi-Wing) across AB-UPT,
      UPT, Transformer, and Transolver architectures.

   .. grid-item-card:: :ph:`thermometer-hot` Fluid Heat Transfer
      :link: noether/heat_transfer
      :link-type: doc
      :class-card: sd-h-100 sd-shadow-sm sd-rounded-3
      :class-title: sd-fs-5 sd-fw-bold sd-text-primary
      :class-body: sd-p-4

      Neural surrogates for heat transfer on the SIMSHIFT Heatsink benchmark - predict 3D velocity, temperature,
      and pressure fields from heatsink geometry.

.. grid:: 1
   :gutter: 2
   :class-container: sd-mt-2

   .. grid-item-card:: :ph:`books` Browse the full Noether Recipe Zoo →
      :link: noether/recipe_zoo
      :link-type: doc
      :class-card: sd-shadow-sm sd-rounded-3
      :class-title: sd-fs-6 sd-fw-bold sd-text-primary sd-mb-0
      :class-body: sd-py-3 sd-px-4
