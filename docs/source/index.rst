Noether Framework Documentation
===============================

Welcome to the Noether Framework documentation. Here you will find available APIs, CLIs, etc.

.. grid:: 1
   :gutter: 2
   :class-container: sd-mb-4

   .. grid-item-card:: 🚀 Start Here: Introduction to Noether
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


.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/prerequisites
   tutorials/getting_started_install_and_verify
   tutorials/training_first_model_with_configs
   tutorials/training_first_model_with_code
   .. tutorials/running_first_inference
   


.. toctree::
   :maxdepth: 1
   :caption: How-to Guides
   :hidden:

   guides/hardware_setup
   guides/linux_cuda_setup
   guides/working_with_cli

   .. guides/data/how_to_load_custom_dataset

   guides/data/how_to_use_private_data_source
   guides/data/how_to_make_a_custom_dataset
   guides/data/how_to_write_a_multistage_pipeline
   guides/data/how_to_write_a_sample_processor

   guides/training/implementing_a_custom_model
   guides/training/implement_a_custom_trainer
   guides/training/use_callbacks

   .. guides/data/how_to_write_data_preprocessors
   .. guides/data/how_to_write_data_collators

   .. guides/training/how_to_configure_all_training_options
   .. guides/training/how_to_use_custom_model
   .. guides/training/how_to_write_custom_loss_or_metric

   guides/inference/how_to_run_evaluation_on_trained_models
   .. guides/inference/how_to_run_inference_via_cli
   .. guides/inference/how_to_run_inference_via_code


.. toctree::
   :maxdepth: 1
   :caption: Noether Zoos
   :hidden:

   zoo/dataset_zoo
   zoo/model_zoo


.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   noether/io/caching
   ../autoapi/noether/index


.. grid:: 1 2 2 2
   :gutter: 2
   :class-container: sd-equal-height

   .. grid-item::
      .. card:: 🧠️ The Noether Framework
        :class-card: sd-h-100
        :link: noether/index
        :link-type: doc
        :shadow: md

        Go deep directly. Learn about our core architecture, key concepts, and the design principles behind the framework.

   .. grid-item::
      .. card:: 🎓 Tutorials
        :class-card: sd-h-100
        :link: tutorials/index
        :link-type: doc
        :shadow: md

        Guided lessons to install the framework, run your first simulation, and understand the basics.

   .. grid-item::
      .. card:: 🛠️ How-to Guides
        :class-card: sd-h-100
        :link: guides/index
        :link-type: doc
        :shadow: md

        Step-by-step recipes for common problems, like loading custom data, using private sources, or writing your
        own data collators.

   .. grid-item::
      .. card:: 📚 Reference
        :class-card: sd-h-100
        :link: ../autoapi/noether/index
        :link-type: doc
        :shadow: md

        Technical lookup for complete API documentation
